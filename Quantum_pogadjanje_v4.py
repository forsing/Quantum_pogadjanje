#!/usr/bin/env python3

# Quantum_pogadjanje


"""
QUANTUM POGADJANJE V4 — Quantum pyramid — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Preserved V4 information flow:
    Alice receives two preceding occupancy vectors;
    a pairwise pyramid compresses them to one binary message;
    Bob has two query-dependent quantum decoders;
    each decoder defines a normalized seven-label distribution;
    the root Born probabilities weight those two distributions.

Quantum replacement:
    leaf RY/RZ state preparation;
    node RY(left), RY(right), CX(right,left), RZ(left);
    retain the left qubit and trace out the right;
    root RX/RY followed by computational-basis measurement;
    Bob uses message-controlled one-qubit preparation/readout.

Training propagates the exact reduced quantum states through
their Bloch representation. These equations are the exact
channels of the stated gates, not tanh neural layers.

Qiskit executes the decoder circuits during every objective call.
Self-checks compare the pyramid with Qiskit density-matrix
simulation and a small full statevector simulation.
Final inference executes the complete compressed pyramid in Qiskit.

No dense 39-qubit statevector is allocated.

No PR or superquantum resources are claimed.
This is a quantum-circuit adaptation of the pyramidal architecture.

No classifier, random initialization, sampling, injected noise,
row shuffling, rounding, duplicate repair or cosmetic filters.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    partial_trace,
)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)

NUMBER_COUNT = 39
DRAW_SIZE = 7

RIDGES = (0.03, 0.3, 3.0)

OPTIMIZER_STEPS = 500
GRADIENT_LIMIT = 5e-5

LEAF_PHASE = math.pi / 4.0

CHECKPOINT_VERSION = "QISKIT_PYRAMID_V4_20261029"

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "Quantum_pogadjanje_v4_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_log(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    require(
        np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0),
        "Neispravne verovatnoće za logaritam.",
    )

    result = np.full_like(probabilities, -np.inf)
    positive = probabilities > 0
    result[positive] = np.log(probabilities[positive])

    return result


def fixed_size_partition(fields, k, with_marginals=False):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 2
        and np.all(np.isfinite(fields))
        and 1 <= k <= fields.shape[1],
        "Neispravan ulaz uslovne normalizacije.",
    )

    n, m = fields.shape

    forward = np.full((n, m + 1, k + 1), -np.inf)
    forward[:, 0, 0] = 0.0

    for number in range(m):
        forward[:, number + 1, 0] = 0.0

        forward[:, number + 1, 1:] = np.logaddexp(
            forward[:, number, 1:],
            fields[:, number, None]
            + forward[:, number, :-1],
        )

    log_z = forward[:, m, k]

    require(
        np.all(np.isfinite(log_z)),
        "Normalizaciona konstanta nije konačna.",
    )

    if not with_marginals:
        return log_z, None

    backward = np.full_like(forward, -np.inf)
    backward[:, m, 0] = 0.0

    for number in range(m - 1, -1, -1):
        backward[:, number, 0] = 0.0

        backward[:, number, 1:] = np.logaddexp(
            backward[:, number + 1, 1:],
            fields[:, number, None]
            + backward[:, number + 1, :-1],
        )

    require(
        np.max(np.abs(
            backward[:, 0, k] - log_z
        )) < 1e-8,
        "Prednja i zadnja normalizacija nisu saglasne.",
    )

    marginal = np.empty_like(fields)

    for number in range(m):
        terms = (
            forward[:, number, :k]
            + backward[:, number + 1, k - 1::-1]
        )

        marginal[:, number] = np.exp(
            fields[:, number]
            + logsumexp(terms, axis=1)
            - log_z
        )

    require(
        np.all(np.isfinite(marginal))
        and np.min(marginal) >= -1e-9
        and np.max(marginal) <= 1.0 + 1e-9,
        "Uslovne marginalne verovatnoće nisu ispravne.",
    )

    require(
        np.max(np.abs(
            marginal.sum(axis=1) - k
        )) < 1e-7,
        "Zbir uslovnih margina nije veličina kombinacije.",
    )

    return log_z, marginal


class QuantumPyramid:
    def __init__(self, m=39, k=7):
        require(
            1 <= k <= m,
            "Neispravne dimenzije piramide.",
        )

        self.m = int(m)
        self.k = int(k)

        self.levels = []
        width = self.m

        while width > 1:
            pairs = width // 2
            self.levels.append((width, pairs))
            width = pairs + width % 2

        self.node_count = sum(
            pairs for _, pairs in self.levels
        )

        self.size = (
            3 * self.m
            + 3 * self.node_count
            + 2
            + 2 * self.m
        )

    def initial(self):
        parameters = np.zeros(self.size)
        position = 0

        labels = np.arange(1, self.m + 1, dtype=float)

        leaf = parameters[
            position:position + 3 * self.m
        ].reshape(3, self.m)

        leaf[0] = 1.1 + 0.02 * np.sin(labels)
        leaf[1] = 0.20 * np.cos(labels * 0.71)
        leaf[2] = 0.15 * np.sin(labels * 0.53)

        position += 3 * self.m

        for _, pairs in self.levels:
            weights = parameters[
                position:position + 3 * pairs
            ].reshape(pairs, 3)

            weights[:, 0] = 0.15
            weights[:, 1] = 0.15
            weights[:, 2] = 0.10

            position += 3 * pairs

        parameters[position:position + 2] = (0.10, 0.15)
        position += 2

        decoder = parameters[position:].reshape(2, self.m)

        # Both branches use the same initial RY angle.
        # Their different input message states give opposite biases.
        decoder[:] = (
            math.pi / 2.0
            + 0.04 * np.sin(labels * 0.43)
        )[None, :]

        return parameters

    def unpack(self, parameters):
        parameters = np.asarray(parameters, dtype=float)

        require(
            parameters.shape == (self.size,)
            and np.all(np.isfinite(parameters)),
            "Neispravni parametri piramide.",
        )

        position = 0

        leaf = parameters[
            position:position + 3 * self.m
        ].reshape(3, self.m)

        position += 3 * self.m
        nodes = []

        for _, pairs in self.levels:
            nodes.append(
                parameters[
                    position:position + 3 * pairs
                ].reshape(pairs, 3)
            )

            position += 3 * pairs

        root = parameters[position:position + 2]
        position += 2

        decoder = parameters[position:].reshape(2, self.m)

        return leaf, nodes, root, decoder

    def encode(self, parameters, x):
        leaf, nodes, root_angles, _ = self.unpack(parameters)

        require(
            x.ndim == 3
            and x.shape[1:] == (2, self.m)
            and np.all(np.isfinite(x)),
            "Neispravan istorijski ulaz piramide.",
        )

        theta = (
            leaf[0][None, :]
            + x[:, 0, :] * leaf[1][None, :]
            + x[:, 1, :] * leaf[2][None, :]
        )

        cp = math.cos(LEAF_PHASE)
        sp = math.sin(LEAF_PHASE)

        current = np.stack(
            (
                cp * np.sin(theta),
                sp * np.sin(theta),
                np.cos(theta),
            ),
            axis=-1,
        )

        traces = []

        for (width, pairs), weights in zip(self.levels, nodes):
            left = current[:, :2 * pairs:2, :]
            right = current[:, 1:2 * pairs:2, :]

            a = weights[:, 0]
            b = weights[:, 1]
            c = weights[:, 2]

            ca, sa = np.cos(a), np.sin(a)
            cb, sb = np.cos(b), np.sin(b)
            cc, sc = np.cos(c), np.sin(c)

            lx = ca * left[:, :, 0] + sa * left[:, :, 2]
            ly = left[:, :, 1]
            lz = -sa * left[:, :, 0] + ca * left[:, :, 2]

            rz = -sb * right[:, :, 0] + cb * right[:, :, 2]

            # Exact reduced state after CX(right,left).
            # Trace out right, then apply RZ(c) to left.
            paired = np.stack(
                (
                    cc * lx - sc * ly * rz,
                    sc * lx + cc * ly * rz,
                    lz * rz,
                ),
                axis=-1,
            )

            require(
                np.max(np.sum(paired * paired, axis=-1)) <= 1.0 + 1e-9,
                "Piramidalni čvor nema fizičku matricu gustine.",
            )

            traces.append(
                (current, paired, lx, ly, lz, rz)
            )

            if width % 2:
                current = np.concatenate(
                    (paired, current[:, -1:, :]),
                    axis=1,
                )
            else:
                current = paired

        root_vector = current[:, 0, :]

        a, b = root_angles
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)

        rx_y = ca * root_vector[:, 1] - sa * root_vector[:, 2]
        rx_z = sa * root_vector[:, 1] + ca * root_vector[:, 2]

        final_z = -sb * root_vector[:, 0] + cb * rx_z

        probabilities = np.column_stack(
            (
                (1.0 + final_z) / 2.0,
                (1.0 - final_z) / 2.0,
            )
        )

        require(
            np.all(np.isfinite(probabilities))
            and np.all(probabilities >= 0)
            and np.all(probabilities <= 1)
            and np.max(np.abs(
                probabilities.sum(axis=1) - 1.0
            )) < 1e-10,
            "Neispravne Born verovatnoće korenske poruke.",
        )

        cache = (
            theta,
            traces,
            root_vector,
            rx_y,
            rx_z,
        )

        return probabilities, cache

    def encoder_gradient(self, parameters, x, cache, dprobability):
        _, nodes, root_angles, _ = self.unpack(parameters)

        theta, traces, root_vector, rx_y, rx_z = cache

        a, b = root_angles
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)

        gz = (
            dprobability[:, 0] - dprobability[:, 1]
        ) / 2.0

        root_gradient = np.array(
            (
                np.sum(gz * cb * rx_y),
                np.sum(
                    gz
                    * (-cb * root_vector[:, 0] - sb * rx_z)
                ),
            )
        )

        upstream = np.stack(
            (
                -sb * gz,
                cb * sa * gz,
                cb * ca * gz,
            ),
            axis=-1,
        )[:, None, :]

        node_gradients = [None] * len(self.levels)

        for level in range(len(self.levels) - 1, -1, -1):
            width, pairs = self.levels[level]
            weights = nodes[level]

            previous, paired, lx, ly, lz, rz = traces[level]

            left = previous[:, :2 * pairs:2, :]
            right = previous[:, 1:2 * pairs:2, :]

            a = weights[:, 0]
            b = weights[:, 1]
            c = weights[:, 2]

            ca, sa = np.cos(a), np.sin(a)
            cb, sb = np.cos(b), np.sin(b)
            cc, sc = np.cos(c), np.sin(c)

            g = upstream[:, :pairs, :]

            px = cc * g[:, :, 0] + sc * g[:, :, 1]
            py = -sc * g[:, :, 0] + cc * g[:, :, 1]
            pz = g[:, :, 2]

            glx = px
            gly = py * rz
            glz = pz * rz

            grz = py * ly + pz * lz

            node_gradients[level] = np.column_stack(
                (
                    np.sum(glx * lz - glz * lx, axis=0),
                    np.sum(
                        grz
                        * (
                            -cb * right[:, :, 0]
                            - sb * right[:, :, 2]
                        ),
                        axis=0,
                    ),
                    np.sum(
                        -g[:, :, 0] * paired[:, :, 1]
                        + g[:, :, 1] * paired[:, :, 0],
                        axis=0,
                    ),
                )
            )

            preceding = np.zeros_like(previous)

            preceding[:, :2 * pairs:2, :] = np.stack(
                (
                    ca * glx - sa * glz,
                    gly,
                    sa * glx + ca * glz,
                ),
                axis=-1,
            )

            preceding[:, 1:2 * pairs:2, :] = np.stack(
                (
                    -sb * grz,
                    np.zeros_like(grz),
                    cb * grz,
                ),
                axis=-1,
            )

            if width % 2:
                preceding[:, -1, :] = upstream[:, -1, :]

            upstream = preceding

        cp = math.cos(LEAF_PHASE)
        sp = math.sin(LEAF_PHASE)

        derivative = np.stack(
            (
                cp * np.cos(theta),
                sp * np.cos(theta),
                -np.sin(theta),
            ),
            axis=-1,
        )

        dtheta = np.sum(upstream * derivative, axis=-1)

        leaf_gradient = np.vstack(
            (
                dtheta.sum(axis=0),
                np.sum(dtheta * x[:, 0, :], axis=0),
                np.sum(dtheta * x[:, 1, :], axis=0),
            )
        )

        parts = [leaf_gradient.ravel()]
        parts.extend(value.ravel() for value in node_gradients)
        parts.append(root_gradient)

        return np.concatenate(parts)

    def decoder(self, parameters):
        _, _, _, angles = self.unpack(parameters)

        q = np.empty((2, self.m))
        r = np.empty_like(q)

        for message in range(2):
            for number in range(self.m):
                circuit = QuantumCircuit(1)

                if message:
                    circuit.x(0)

                circuit.ry(float(angles[message, number]), 0)
                circuit.rz(LEAF_PHASE, 0)

                state = Statevector.from_instruction(circuit)
                probabilities = np.abs(state.data) ** 2

                require(
                    abs(float(probabilities.sum()) - 1.0) < 1e-10,
                    "Bobov Qiskit dekoder nije normalizovan.",
                )

                r[message, number] = probabilities[0]
                q[message, number] = probabilities[1]

        require(
            np.all(q > 0)
            and np.all(r > 0)
            and np.all(np.isfinite(q))
            and np.all(np.isfinite(r)),
            "Nulta ili nevažeća Born verovatnoća dekodera.",
        )

        derivative = np.stack(
            (
                0.5 * np.sin(angles[0]),
                -0.5 * np.sin(angles[1]),
            )
        )

        return q, r, derivative

    @staticmethod
    def node_circuit(weights):
        circuit = QuantumCircuit(2)
        circuit.ry(float(weights[0]), 0)
        circuit.ry(float(weights[1]), 1)
        circuit.cx(1, 0)
        circuit.rz(float(weights[2]), 0)
        return circuit

    def explicit_message(self, parameters, x_row):
        leaf, nodes, root, _ = self.unpack(parameters)

        theta = (
            leaf[0]
            + x_row[0] * leaf[1]
            + x_row[1] * leaf[2]
        )

        active = []

        for angle in theta:
            circuit = QuantumCircuit(1)
            circuit.ry(float(angle), 0)
            circuit.rz(LEAF_PHASE, 0)

            active.append(
                DensityMatrix(
                    Statevector.from_instruction(circuit)
                )
            )

        for (width, pairs), weights in zip(self.levels, nodes):
            following = []

            for pair in range(pairs):
                left = active[2 * pair]
                right = active[2 * pair + 1]

                joint = DensityMatrix(
                    np.kron(right.data, left.data)
                )

                evolved = joint.evolve(
                    self.node_circuit(weights[pair])
                )

                following.append(
                    partial_trace(evolved, [1])
                )

            if width % 2:
                following.append(active[-1])

            active = following

        circuit = QuantumCircuit(1)
        circuit.rx(float(root[0]), 0)
        circuit.ry(float(root[1]), 0)

        final_state = active[0].evolve(circuit)
        probabilities = np.asarray(
            final_state.probabilities(),
            dtype=float,
        )

        reference = self.encode(
            parameters,
            np.asarray(x_row)[None, :, :],
        )[0][0]

        error = float(
            np.max(np.abs(probabilities - reference))
        )

        require(
            error < 2e-9,
            f"Piramida nije saglasna sa Qiskitom: {error:.3e}",
        )

        return probabilities, error

    def full_small_message(self, parameters, x_row):
        require(
            self.m <= 8,
            "Puna simulacija dozvoljena je samo za malu samoproveru.",
        )

        leaf, nodes, root, _ = self.unpack(parameters)

        theta = (
            leaf[0]
            + x_row[0] * leaf[1]
            + x_row[1] * leaf[2]
        )

        circuit = QuantumCircuit(self.m)

        for qubit, angle in enumerate(theta):
            circuit.ry(float(angle), qubit)
            circuit.rz(LEAF_PHASE, qubit)

        active = list(range(self.m))

        for (width, pairs), weights in zip(self.levels, nodes):
            following = []

            for pair in range(pairs):
                left = active[2 * pair]
                right = active[2 * pair + 1]

                circuit.compose(
                    self.node_circuit(weights[pair]),
                    qubits=[left, right],
                    inplace=True,
                )

                following.append(left)

            if width % 2:
                following.append(active[-1])

            active = following

        circuit.rx(float(root[0]), active[0])
        circuit.ry(float(root[1]), active[0])

        return Statevector.from_instruction(circuit).probabilities(
            qargs=[active[0]]
        )


def distribution_terms(model, parameters, x, targets=None, gradient=False):
    weights, cache = model.encode(parameters, x)
    q, r, decoder_derivative = model.decoder(parameters)

    fields = np.log(q) - np.log(r)

    log_z, marginal = fixed_size_partition(
        fields,
        model.k,
        with_marginals=gradient,
    )

    if targets is None:
        return (
            weights, fields, log_z, marginal,
            cache, q, r, decoder_derivative,
        )

    component_logp = targets @ fields.T - log_z[None, :]
    joint = safe_log(weights) + component_logp
    logp = logsumexp(joint, axis=1)

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća kompletne kombinacije.",
    )

    responsibility = np.exp(joint - logp[:, None])

    return (
        weights, cache, q, r, decoder_derivative,
        fields, log_z, marginal, component_logp,
        logp, responsibility,
    )


def objective(parameters, model, x, targets, ridge, gradient=True):
    (
        weights, cache, q, r, decoder_derivative,
        fields, log_z, marginal, component_logp,
        logp, responsibility,
    ) = distribution_terms(
        model,
        parameters,
        x,
        targets,
        gradient=gradient,
    )

    difference = parameters - model.initial()

    loss = (
        -float(logp.mean())
        + 0.5 * ridge * float(np.dot(difference, difference))
    )

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    n = len(targets)

    # Derivatives remain defined even if one root weight is zero.
    dweight = -np.exp(
        component_logp - logp[:, None]
    ) / n

    encoder_gradient = model.encoder_gradient(
        parameters,
        x,
        cache,
        dweight,
    )

    field_gradient = (
        responsibility.sum(axis=0)[:, None] * marginal
        - responsibility.T @ targets
    ) / n

    decoder_gradient = (
        field_gradient
        * decoder_derivative
        / (q * r)
    )

    result = np.concatenate(
        (encoder_gradient, decoder_gradient.ravel())
    )

    result += ridge * difference

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


def exact_mixture_maximum(fields, offsets, k):
    """
    Exact supported-set search for a mixture of two exponential
    fixed-size distributions.

    At a global maximum, the selected set also maximizes a positive
    weighted sum of the two component scores.

    All possible top-k sets occur between pairwise score crossings.
    Fractions preserve the exact ordering of the supplied float fields.
    No random search or cosmetic filtering is used.
    """
    fields = np.asarray(fields, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    require(
        fields.ndim == 2
        and fields.shape[0] == 2
        and np.all(np.isfinite(fields))
        and offsets.shape == (2,)
        and not np.any(np.isnan(offsets))
        and not np.any(np.isposinf(offsets))
        and 1 <= k <= fields.shape[1],
        "Neispravan ulaz maksimuma mešavine.",
    )

    active = np.flatnonzero(np.isfinite(offsets))

    require(len(active) > 0, "Obe grane imaju nultu težinu.")

    m = fields.shape[1]

    if len(active) == 1:
        branch = int(active[0])

        order = sorted(
            range(m),
            key=lambda i: (-float(fields[branch, i]), i),
        )

        best = tuple(sorted(order[:k]))
        score = float(
            offsets[branch] + fields[branch, list(best)].sum()
        )

        return best, score, 1

    h0 = [Fraction(float(value)) for value in fields[0]]
    h1 = [Fraction(float(value)) for value in fields[1]]
    slopes = [a - b for a, b in zip(h0, h1)]

    boundaries = {Fraction(0), Fraction(1)}

    for first in range(m):
        for second in range(first + 1, m):
            denominator = slopes[first] - slopes[second]

            if denominator == 0:
                continue

            crossing = (
                h1[second] - h1[first]
            ) / denominator

            if 0 < crossing < 1:
                boundaries.add(crossing)

    boundaries = sorted(boundaries)

    probes = list(boundaries)

    probes.extend(
        (left + right) / 2
        for left, right in zip(boundaries, boundaries[1:])
    )

    candidates = set()

    for weight in probes:
        linear_scores = [
            h1[i] + weight * slopes[i]
            for i in range(m)
        ]

        order = sorted(
            range(m),
            key=lambda i: (-linear_scores[i], i),
        )

        candidates.add(tuple(sorted(order[:k])))

    best = None
    best_score = -math.inf

    for selected in sorted(candidates):
        component_scores = (
            offsets
            + fields[:, list(selected)].sum(axis=1)
        )

        score = float(logsumexp(component_scores))

        if score > best_score:
            best_score = score
            best = selected

    require(
        best is not None and np.isfinite(best_score),
        "Maksimum mešavine nije pronađen.",
    )

    return best, best_score, len(candidates)


def contexts(occupancy, targets):
    targets = np.asarray(targets, dtype=np.int64)

    require(
        targets.ndim == 1
        and np.all(targets >= 1)
        and np.all(targets <= len(occupancy)),
        "Neispravni indeksi istorijskih konteksta.",
    )

    m = occupancy.shape[1]
    center = DRAW_SIZE / m

    result = np.zeros((len(targets), 2, m))

    for lag in range(2):
        previous = targets - lag - 1
        valid = previous >= 0

        result[valid, lag, :] = (
            occupancy[previous[valid]] - center
        )

    return result


def load_draws(path):
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = []
    blank = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                blank += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = tuple(int(item.strip()) for item in row)
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == DRAW_SIZE
                and min(numbers) >= 1
                and max(numbers) <= NUMBER_COUNT,
                f"{path.name}, red {line_number}: neispravna sedmorka.",
            )

            rows.append(
                tuple(value - 1 for value in sorted(numbers))
            )

    require(len(rows) >= 100, "Nedovoljno istorijskih kola.")

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros((len(draws), NUMBER_COUNT))
    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def checkpoint_path(x, targets, ridge):
    digest = hashlib.sha256()

    digest.update(repr(
        (
            CHECKPOINT_VERSION,
            float(ridge),
            OPTIMIZER_STEPS,
            qiskit.__version__,
            np.__version__,
        )
    ).encode("utf-8"))

    for array in (x, targets):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())

    return CHECKPOINT_DIR / (digest.hexdigest() + ".npz")


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                parameters=np.asarray(parameters, dtype=float),
                iterations=np.array(int(iterations)),
                done=np.array(int(done)),
                message=np.array(str(message)),
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, shape):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            parameters = saved["parameters"].astype(float)
            iterations = int(saved["iterations"].item())
            done = bool(saved["done"].item())
            message = str(saved["message"].item())

        require(
            parameters.shape == shape
            and np.all(np.isfinite(parameters))
            and iterations >= 0,
            "Neispravna kontrolna tačka.",
        )

        return parameters, iterations, done, message

    except (
        OSError, ValueError, KeyError, RuntimeError,
        EOFError, BadZipFile,
    ) as error:
        say(
            f"Kontrolna tačka nije učitana: {path.name}",
            str(error),
        )

        return None


@dataclass
class Fitted:
    parameters: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    message: str


def fit(model, x, targets, ridge, label):
    initial = model.initial()
    path = checkpoint_path(x, targets, ridge)
    saved = load_checkpoint(path, initial.shape)

    parameters = initial.copy()
    used_iterations = 0
    done = False
    message = "Novi proračun."

    if saved is not None:
        parameters, used_iterations, done, message = saved

    started = time.perf_counter()

    say(
        "",
        label,
        f"λ={ridge:g} | prelaza={len(targets)}"
        f" | kvantnih parametara={model.size}",
    )

    try:
        best_loss, best_gradient = objective(
            parameters, model, x, targets, ridge
        )

    except (
        RuntimeError, FloatingPointError, np.linalg.LinAlgError,
    ):
        require(
            saved is not None,
            "Početna kvantna raspodela nije numerički validna.",
        )

        parameters = initial.copy()
        used_iterations = 0
        done = False

        best_loss, best_gradient = objective(
            parameters, model, x, targets, ridge
        )

    best_parameters = parameters.copy()
    best_gradient = best_gradient.copy()

    added_iterations = 0
    optimizer_success = False

    if done:
        say("Učitan završen proračun iz kontrolne tačke.")

    else:
        save_checkpoint(
            path,
            best_parameters,
            used_iterations,
            False,
            "Početna ili nastavljena validna tačka.",
        )

        def function(current):
            nonlocal best_loss
            nonlocal best_parameters
            nonlocal best_gradient

            loss, gradient = objective(
                current, model, x, targets, ridge
            )

            if loss < best_loss:
                best_loss = float(loss)
                best_parameters = current.copy()
                best_gradient = gradient.copy()

            return loss, gradient

        def callback(current):
            nonlocal added_iterations
            added_iterations += 1

            if added_iterations % 25 == 0:
                total = used_iterations + added_iterations

                save_checkpoint(
                    path,
                    best_parameters,
                    total,
                    False,
                    "Sačuvana najbolja validna tačka.",
                )

                say(
                    f"Iteracija {total}"
                    f" | proteklo {time.perf_counter() - started:.1f} s",
                    f"Najbolji cilj: {best_loss:.9f}",
                    "Max|grad| te tačke:"
                    f" {np.max(np.abs(best_gradient)):.3e}",
                )

        remaining = max(0, OPTIMIZER_STEPS - used_iterations)

        if remaining:
            try:
                result = minimize(
                    function,
                    parameters,
                    method="L-BFGS-B",
                    jac=True,
                    callback=callback,
                    options={
                        "maxiter": remaining,
                        "maxls": 60,
                        "maxcor": 20,
                        "ftol": 1e-13,
                        "gtol": 1e-6,
                    },
                )

                optimizer_success = bool(result.success)
                message = str(result.message)
                function(result.x)

            except (
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                message = (
                    "Pretraga prekinuta; zadržana validna tačka: "
                    + str(error)
                )

        else:
            message = "Dostignut sačuvani budžet iteracija."

        used_iterations += added_iterations

    checked_loss, checked_gradient = objective(
        best_parameters,
        model,
        x,
        targets,
        ridge,
    )

    maximum_gradient = float(
        np.max(np.abs(checked_gradient))
    )

    # Explicit Qiskit check before accepting the fitted parameters.
    model.explicit_message(best_parameters, x[0])

    converged = maximum_gradient <= GRADIENT_LIMIT

    if not done:
        prefix = (
            "OPTIMIZER_SUCCESS"
            if optimizer_success
            else "OPTIMIZER_NOT_CONFIRMED"
        )

        message = prefix + ": " + message

        save_checkpoint(
            path,
            best_parameters,
            used_iterations,
            True,
            message,
        )

    say(
        f"Zadržana validna tačka: {checked_loss:.9f}",
        f"Max|grad|: {maximum_gradient:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if converged else "NIJE dostignut"),
        message,
        f"Trajanje ovog poziva: {time.perf_counter() - started:.1f} s",
    )

    return Fitted(
        parameters=best_parameters.copy(),
        loss=float(checked_loss),
        gradient_norm=maximum_gradient,
        converged=converged,
        message=message,
    )


def evaluate_nll(model, fitted, x, targets):
    result = distribution_terms(
        model,
        fitted.parameters,
        x,
        targets,
    )

    return -float(result[-2].mean())


def self_checks():
    say(
        "",
        "Pokrećem ugrađene Qiskit i matematičke provere.",
    )

    started = time.perf_counter()

    model = QuantumPyramid(5, 2)
    parameters = model.initial()

    x = np.array(
        [
            [[0.6, -0.4, 0.6, -0.4, -0.4],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[-0.4, 0.6, -0.4, 0.6, -0.4],
             [0.6, -0.4, 0.6, -0.4, -0.4]],
        ],
        dtype=float,
    )

    targets = np.array(
        [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
        dtype=float,
    )

    compact, _ = model.encode(parameters, x)
    largest_circuit_error = 0.0

    for sample in range(len(x)):
        explicit, error = model.explicit_message(
            parameters,
            x[sample],
        )

        full = model.full_small_message(
            parameters,
            x[sample],
        )

        require(
            np.max(np.abs(explicit - full)) < 2e-9
            and np.max(np.abs(compact[sample] - full)) < 2e-9,
            "Puna mala Qiskit piramida nije prošla proveru.",
        )

        largest_circuit_error = max(
            largest_circuit_error,
            error,
        )

    ridge = 0.03

    _, analytic = objective(
        parameters,
        model,
        x,
        targets,
        ridge,
    )

    epsilon = 1e-6
    numerical = np.empty_like(parameters)

    for parameter in range(len(parameters)):
        positive = parameters.copy()
        negative = parameters.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        numerical[parameter] = (
            objective(
                positive, model, x, targets, ridge,
                gradient=False,
            )
            - objective(
                negative, model, x, targets, ridge,
                gradient=False,
            )
        ) / (2.0 * epsilon)

    gradient_error = float(
        np.max(np.abs(analytic - numerical))
    )

    require(
        gradient_error < 3e-6,
        f"Provera gradijenta nije prošla: {gradient_error:.3e}",
    )

    terms = distribution_terms(
        model,
        parameters,
        x,
    )

    weights, fields, log_z = terms[:3]

    subsets = list(combinations(range(model.m), model.k))
    selected = np.asarray(subsets, dtype=np.int64)

    component = (
        fields[:, selected].sum(axis=2).T
        - log_z[None, :]
    )

    for sample in range(len(x)):
        log_probabilities = logsumexp(
            component + safe_log(weights[sample])[None, :],
            axis=1,
        )

        require(
            abs(float(np.exp(log_probabilities).sum()) - 1.0) < 1e-11,
            "Potpuna provera normalizacije mešavine nije prošla.",
        )

        best, score, _ = exact_mixture_maximum(
            fields,
            safe_log(weights[sample]) - log_z,
            model.k,
        )

        require(
            best == subsets[int(np.argmax(log_probabilities))]
            and abs(score - float(np.max(log_probabilities))) < 1e-10,
            "Provera maksimuma mešavine nije prošla.",
        )

    # Independent deterministic score geometries.
    for case in range(1, 13):
        h = np.array(
            [
                [math.sin((i + 1) * case * 0.37) for i in range(6)],
                [math.cos((i + 2) * case * 0.29) for i in range(6)],
            ]
        )

        offsets = np.log(np.array([0.37, 0.63]))
        sets = list(combinations(range(6), 2))
        indices = np.asarray(sets, dtype=np.int64)

        exhaustive = logsumexp(
            h[:, indices].sum(axis=2).T + offsets,
            axis=1,
        )

        best, score, _ = exact_mixture_maximum(h, offsets, 2)

        require(
            best == sets[int(np.argmax(exhaustive))]
            and abs(score - float(np.max(exhaustive))) < 1e-10,
            "Nezavisna provera maksimuma nije prošla.",
        )

    for k in (1, 5):
        z, marginal = fixed_size_partition(
            np.zeros((1, 5)),
            k,
            with_marginals=True,
        )

        require(
            abs(z[0] - math.log(math.comb(5, k))) < 1e-11
            and np.max(np.abs(marginal[0] - k / 5)) < 1e-11,
            "Granična provera normalizacije nije prošla.",
        )

    say(
        "Ugrađene provere su prošle.",
        f"Greška piramide prema Qiskitu: {largest_circuit_error:.3e}",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
        "Provereni su mala puna simulacija, normalizacija i maksimum.",
        f"Trajanje provera: {time.perf_counter() - started:.1f} s",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def print_result(result):
    numbers = " ".join(
        f"{value:02d}" for value in result["next"]
    )

    say(
        "",
        result["name"],
        f"NEXT: {numbers}",
        f"Validacioni NLL: {result['validation']:.9f}",
        f"Odvojeni test NLL: {result['test']:.9f}",
        f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
        "Korenske Born verovatnoće: "
        + " ".join(f"{value:.9f}" for value in result["weights"]),
        f"Zbir uslovnih margina: {result['marginal_sum']:.12f}",
        f"Proverenih kandidata za maksimum: {result['candidate_count']}",
        f"Greška piramide prema Qiskitu: {result['circuit_error']:.3e}",
        f"Osetljivost uslovnih margina: {result['sensitivity']:.9e}",
        f"Max|grad| završnog modela: {result['gradient']:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if result["converged"] else "NIJE dostignut"),
    )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process(path, draws, occupancy):
    model = QuantumPyramid(NUMBER_COUNT, DRAW_SIZE)

    x = contexts(
        occupancy,
        np.arange(1, len(occupancy)),
    )

    targets = occupancy[1:].copy()
    n = len(targets)

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno istorije za hronološku podelu.",
    )

    scores = []

    for ridge in RIDGES:
        total_nll = 0.0
        total_count = 0
        incomplete_folds = 0

        for fold, (train_end, validation_end) in enumerate(
            ((split60, split70), (split70, split80)),
            1,
        ):
            fitted = fit(
                model,
                x[:train_end],
                targets[:train_end],
                ridge,
                f"{path.name} / validacija {fold}",
            )

            incomplete_folds += int(not fitted.converged)

            score = evaluate_nll(
                model,
                fitted,
                x[train_end:validation_end],
                targets[train_end:validation_end],
            )

            count = validation_end - train_end
            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count
        scores.append((score, ridge, incomplete_folds))

        say(
            f"KANDIDAT: λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
            f"Podela bez potvrđenog gradijentnog uslova: {incomplete_folds}",
        )

    validation_nll, ridge, incomplete_folds = min(
        scores,
        key=lambda item: item[0],
    )

    test_model = fit(
        model,
        x[:split80],
        targets[:split80],
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        model,
        test_model,
        x[split80:],
        targets[split80:],
    )

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"NLL cele sedmorke: {test_nll:.9f}",
    )

    final_model = fit(
        model,
        x,
        targets,
        ridge,
        f"{path.name} / završno učenje svih {n} prelaza",
    )

    next_context = contexts(
        occupancy,
        [len(occupancy)],
    )

    # Direct compressed Qiskit execution supplies final message weights.
    next_weights, circuit_error = model.explicit_message(
        final_model.parameters,
        next_context[0],
    )

    q, r, _ = model.decoder(final_model.parameters)
    fields = np.log(q) - np.log(r)

    log_z, branch_marginals = fixed_size_partition(
        fields,
        DRAW_SIZE,
        with_marginals=True,
    )

    offsets = safe_log(next_weights) - log_z

    best, logp, candidate_count = exact_mixture_maximum(
        fields,
        offsets,
        DRAW_SIZE,
    )

    require(
        len(best) == DRAW_SIZE
        and len(set(best)) == DRAW_SIZE
        and min(best) >= 0
        and max(best) < NUMBER_COUNT,
        "NEXT kombinacija nije legalna.",
    )

    require(
        -math.log(math.comb(NUMBER_COUNT, DRAW_SIZE)) - 1e-7
        <= logp <= 1e-7,
        "Verovatnoća NEXT maksimuma nije ispravna.",
    )

    marginal = next_weights @ branch_marginals

    require(
        abs(float(marginal.sum()) - DRAW_SIZE) < 1e-7,
        "Zbir NEXT margina nije sedam.",
    )

    old_weights, _ = model.encode(
        final_model.parameters,
        x[-8:],
    )

    old_marginals = old_weights @ branch_marginals

    sensitivity = float(
        np.max(np.abs(old_marginals - marginal[None, :]))
    )

    weight_range = float(np.ptp(
        np.concatenate(
            (old_weights[:, 1], [next_weights[1]])
        )
    ))

    warnings = []

    if incomplete_folds:
        warnings.append(
            "Izabrani kandidat ima validacione treninge"
            " bez potvrđenog gradijentnog uslova."
        )

    if not test_model.converged:
        warnings.append(
            "Model odvojenog testa nema potvrđen gradijentni uslov."
        )

    if not final_model.converged:
        warnings.append(
            "Završni model nema potvrđenu konvergenciju;"
            " NEXT je maksimum njegove validne raspodele."
        )

    run = longest_run(best)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if best[-1] - best[0] <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva je u rasponu od deset brojeva."
        )

    repeats = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if repeats:
        warnings.append(
            f"NEXT se pojavljuje {repeats} puta u istoriji."
        )

    if weight_range < 1e-6:
        warnings.append(
            "Korenska kvantna poruka slabo reaguje"
            " na proverene istorijske kontekste."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Uslovne izlazne margine slabo reaguju na istoriju."
        )

    if np.max(np.abs(
        branch_marginals[0] - branch_marginals[1]
    )) < 1e-6:
        warnings.append(
            "Dve izlazne grane imaju gotovo iste uslovne margine."
        )

    result = {
        "name": path.name,
        "next": tuple(value + 1 for value in best),
        "validation": validation_nll,
        "test": test_nll,
        "probability": math.exp(logp),
        "weights": next_weights,
        "marginal_sum": float(marginal.sum()),
        "candidate_count": candidate_count,
        "circuit_error": circuit_error,
        "sensitivity": sensitivity,
        "gradient": final_model.gradient_norm,
        "converged": final_model.converged,
        "warnings": warnings,
    }

    print_result(result)

    return result


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    say(
        "QUANTUM POGADJANJE V4 — Qiskit piramida — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Alice koristi dva prethodna kola.",
        "Kvantna piramida koristi RY, RZ, CNOT i delimični trag.",
        "Korenska RX/RY merenja daju jednu binarnu poruku.",
        "Bobove Born verovatnoće računaju Qiskit kola.",
        "Trening koristi tačnu kompaktnu reprezentaciju kvantnih kanala.",
        "Završna piramida izvršava se preko Qiskit matrica gustine.",
        "Nema tanh piramide niti klasičnog izlaznog regresora.",
        "Istorija menja Born težine dve uslovne izlazne raspodele.",
        "Svaka grana se zasebno normalizuje na legalne sedmorke.",
        "Zatim se grane mešaju korenskim Born verovatnoćama.",
        "Ne koristi se PR niti superkvantni resurs.",
        "Nema slučajnog uzorkovanja, šuma ili mešanja redova.",
        "Ne pravi se gust vektor stanja sa 39 kubita.",
        "Limit iteracija ne odbacuje validnu sačuvanu raspodelu.",
        "Konvergencija se prijavljuje odvojeno.",
        f"Budžet: najviše {OPTIMIZER_STEPS} iteracija po treningu.",
        f"Kontrolne tačke: {CHECKPOINT_DIR}",
    )

    resolved = [path.resolve() for path in FILES]

    require(
        len(set(resolved)) == len(FILES),
        "Ulazne putanje nisu različite.",
    )

    datasets = []
    hashes = []

    for path in FILES:
        require(path.is_file(), f"Fajl nije pronađen: {path}")

        draws, occupancy, fingerprint = load_draws(path)

        datasets.append((path, draws, occupancy))
        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, occupancy in datasets:
        results.append(
            process(path, draws, occupancy)
        )

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if results[first]["next"] == results[second]["next"]:
                results[first]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[second]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

                results[second]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[first]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

    say("", "ZAVRŠNA TRI NEXT REZULTATA")

    for result in results:
        print_result(result)

    say(
        "",
        "Obrada je završena.",
        "NEXT je maksimum mešavine za sačuvane parametre.",
        "Nepotvrđena konvergencija ostaje označena.",
        "Prediktivna vrednost procenjuje se odvojenim testom.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM POGADJANJE V4 — Qiskit piramida — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Alice koristi dva prethodna kola.
Kvantna piramida koristi RY, RZ, CNOT i delimični trag.
Korenska RX/RY merenja daju jednu binarnu poruku.
Bobove Born verovatnoće računaju Qiskit kola.
Trening koristi tačnu kompaktnu reprezentaciju kvantnih kanala.
Završna piramida izvršava se preko Qiskit matrica gustine.
Nema tanh piramide niti klasičnog izlaznog regresora.
Istorija menja Born težine dve uslovne izlazne raspodele.
Svaka grana se zasebno normalizuje na legalne sedmorke.
Zatim se grane mešaju korenskim Born verovatnoćama.
Ne koristi se PR niti superkvantni resurs.
Nema slučajnog uzorkovanja, šuma ili mešanja redova.
Ne pravi se gust vektor stanja sa 39 kubita.
Limit iteracija ne odbacuje validnu sačuvanu raspodelu.
Konvergencija se prijavljuje odvojeno.
Budžet: najviše 500 iteracija po treningu.
Kontrolne tačke: /Users/4c/Desktop/GHQ/Astra/Quantum_pogadjanje/Quantum_pogadjanje_v4_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem ugrađene Qiskit i matematičke provere.
Ugrađene provere su prošle.
Greška piramide prema Qiskitu: 5.551e-16
Najveća greška gradijenta: 3.431e-10
Provereni su mala puna simulacija, normalizacija i maksimum.
Trajanje provera: 0.2 s

loto7_4710_k86.csv / validacija 1
λ=0.03 | prelaza=2825 | kvantnih parametara=311
Iteracija 25 | proteklo 0.6 s
Najbolji cilj: 16.540683468
Max|grad| te tačke: 2.013e-05
Zadržana validna tačka: 16.540683372
Max|grad|: 9.471e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.0 s
Validacioni NLL: 16.554140003

loto7_4710_k86.csv / validacija 2
λ=0.03 | prelaza=3296 | kvantnih parametara=311
Iteracija 25 | proteklo 0.8 s
Najbolji cilj: 16.542021901
Max|grad| te tačke: 6.885e-05
Zadržana validna tačka: 16.542021089
Max|grad|: 9.582e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.3 s
Validacioni NLL: 16.548187773
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.551163888
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
λ=0.3 | prelaza=2825 | kvantnih parametara=311
Zadržana validna tačka: 16.545386340
Max|grad|: 4.651e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
Validacioni NLL: 16.550368972

loto7_4710_k86.csv / validacija 2
λ=0.3 | prelaza=3296 | kvantnih parametara=311
Zadržana validna tačka: 16.545914305
Max|grad|: 3.393e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
Validacioni NLL: 16.548465362
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.549417167
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
λ=3 | prelaza=2825 | kvantnih parametara=311
Zadržana validna tačka: 16.547933728
Max|grad|: 1.386e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.549362776

loto7_4710_k86.csv / validacija 2
λ=3 | prelaza=3296 | kvantnih parametara=311
Zadržana validna tačka: 16.548136874
Max|grad|: 1.401e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.548265848
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.548814312
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / odvojeni test
λ=3 | prelaza=3767 | kvantnih parametara=311
Zadržana validna tačka: 16.548142156
Max|grad|: 1.051e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
ODVOJENI TEST: 942 prelaza
NLL cele sedmorke: 16.547949699

loto7_4710_k86.csv / završno učenje svih 4709 prelaza
λ=3 | prelaza=4709 | kvantnih parametara=311
Zadržana validna tačka: 16.548088008
Max|grad|: 1.382e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s

loto7_4710_k86.csv
NEXT: 10 11 12 24 25 26 39
Validacioni NLL: 16.548814312
Odvojeni test NLL: 16.547949699
Modelna verovatnoća NEXT: 7.81278005119e-08
Korenske Born verovatnoće: 0.459371423 0.540628577
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 18
Greška piramide prema Qiskitu: 1.110e-15
Osetljivost uslovnih margina: 1.771622546e-05
Max|grad| završnog modela: 1.382e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.03 | prelaza=1785 | kvantnih parametara=311
Iteracija 25 | proteklo 0.6 s
Najbolji cilj: 16.532914611
Max|grad| te tačke: 3.784e-04
Zadržana validna tačka: 16.532903700
Max|grad|: 6.502e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.9 s
Validacioni NLL: 16.567846692

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.03 | prelaza=2083 | kvantnih parametara=311
Iteracija 25 | proteklo 0.6 s
Najbolji cilj: 16.536253973
Max|grad| te tačke: 7.783e-04
Iteracija 50 | proteklo 1.1 s
Najbolji cilj: 16.536020476
Max|grad| te tačke: 1.483e-05
Zadržana validna tačka: 16.536020433
Max|grad|: 9.200e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.4 s
Validacioni NLL: 16.558758341
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.563310154
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.3 | prelaza=1785 | kvantnih parametara=311
Zadržana validna tačka: 16.543000383
Max|grad|: 9.749e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.548017765

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.3 | prelaza=2083 | kvantnih parametara=311
Zadržana validna tačka: 16.543417808
Max|grad|: 1.174e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
Validacioni NLL: 16.553686196
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.550847217
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_2977.csv / validacija 1
λ=3 | prelaza=1785 | kvantnih parametara=311
Zadržana validna tačka: 16.547588049
Max|grad|: 2.430e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.548273043

loto7_4710_k86_loto_2977.csv / validacija 2
λ=3 | prelaza=2083 | kvantnih parametara=311
Zadržana validna tačka: 16.547691396
Max|grad|: 1.736e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.550041823
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.549155946
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_2977.csv / odvojeni test
λ=3 | prelaza=2380 | kvantnih parametara=311
Zadržana validna tačka: 16.547985124
Max|grad|: 1.539e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
ODVOJENI TEST: 596 prelaza
NLL cele sedmorke: 16.547976809

loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza
λ=3 | prelaza=2976 | kvantnih parametara=311
Zadržana validna tačka: 16.547956037
Max|grad|: 2.110e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s

loto7_4710_k86_loto_2977.csv
NEXT: 10 11 12 24 25 26 39
Validacioni NLL: 16.549155946
Odvojeni test NLL: 16.547976809
Modelna verovatnoća NEXT: 7.77826909454e-08
Korenske Born verovatnoće: 0.458709037 0.541290963
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 18
Greška piramide prema Qiskitu: 2.776e-16
Osetljivost uslovnih margina: 1.804647919e-05
Max|grad| završnog modela: 2.110e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.03 | prelaza=1039 | kvantnih parametara=311
Iteracija 25 | proteklo 0.5 s
Najbolji cilj: 16.514725977
Max|grad| te tačke: 4.474e-04
Zadržana validna tačka: 16.514702491
Max|grad|: 7.279e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.8 s
Validacioni NLL: 16.569158650

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.03 | prelaza=1212 | kvantnih parametara=311
Iteracija 25 | proteklo 0.5 s
Najbolji cilj: 16.519320642
Max|grad| te tačke: 1.518e-05
Zadržana validna tačka: 16.519320634
Max|grad|: 4.089e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.7 s
Validacioni NLL: 16.572713254
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.570935952
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.3 | prelaza=1039 | kvantnih parametara=311
Zadržana validna tačka: 16.536862782
Max|grad|: 5.740e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.551785577

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.3 | prelaza=1212 | kvantnih parametara=311
Zadržana validna tačka: 16.538848555
Max|grad|: 2.317e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.555678644
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.553732110
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=3 | prelaza=1039 | kvantnih parametara=311
Zadržana validna tačka: 16.546187488
Max|grad|: 9.277e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.548726567

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=3 | prelaza=1212 | kvantnih parametara=311
Zadržana validna tačka: 16.546639479
Max|grad|: 8.073e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
Validacioni NLL: 16.549399339
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.549062953
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
λ=3 | prelaza=1385 | kvantnih parametara=311
Zadržana validna tačka: 16.546951576
Max|grad|: 4.959e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 16.548239526

loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza
λ=3 | prelaza=1732 | kvantnih parametara=311
Zadržana validna tačka: 16.547223829
Max|grad|: 3.468e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.2 s

loto7_4710_k86_loto_plus_1733.csv
NEXT: 10 11 12 25 26 27 39
Validacioni NLL: 16.549062953
Odvojeni test NLL: 16.548239526
Modelna verovatnoća NEXT: 7.9467890837e-08
Korenske Born verovatnoće: 0.459219521 0.540780479
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 16
Greška piramide prema Qiskitu: 1.110e-15
Osetljivost uslovnih margina: 3.900435549e-06
Max|grad| završnog modela: 3.468e-07
Gradijentni uslov: dostignut



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 10 11 12 24 25 26 39
Validacioni NLL: 16.548814312
Odvojeni test NLL: 16.547949699
Modelna verovatnoća NEXT: 7.81278005119e-08
Korenske Born verovatnoće: 0.459371423 0.540628577
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 18
Greška piramide prema Qiskitu: 1.110e-15
Osetljivost uslovnih margina: 1.771622546e-05
Max|grad| završnog modela: 1.382e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Ista NEXT kombinacija kao u loto7_4710_k86_loto_2977.csv; proveriti uticaj istorije i degeneraciju.

loto7_4710_k86_loto_2977.csv
NEXT: 10 11 12 24 25 26 39
Validacioni NLL: 16.549155946
Odvojeni test NLL: 16.547976809
Modelna verovatnoća NEXT: 7.77826909454e-08
Korenske Born verovatnoće: 0.458709037 0.541290963
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 18
Greška piramide prema Qiskitu: 2.776e-16
Osetljivost uslovnih margina: 1.804647919e-05
Max|grad| završnog modela: 2.110e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Ista NEXT kombinacija kao u loto7_4710_k86.csv; proveriti uticaj istorije i degeneraciju.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 10 11 12 25 26 27 39
Validacioni NLL: 16.549062953
Odvojeni test NLL: 16.548239526
Modelna verovatnoća NEXT: 7.9467890837e-08
Korenske Born verovatnoće: 0.459219521 0.540780479
Zbir uslovnih margina: 7.000000000000
Proverenih kandidata za maksimum: 16
Greška piramide prema Qiskitu: 1.110e-15
Osetljivost uslovnih margina: 3.900435549e-06
Max|grad| završnog modela: 3.468e-07
Gradijentni uslov: dostignut

Obrada je završena.
NEXT je maksimum mešavine za sačuvane parametre.
Nepotvrđena konvergencija ostaje označena.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
V4 - piramidalni model sa dve prethodne kombinacije i jednom binarnom porukom. 
Zameniću njegove tanh čvorove kvantnim rotacijama i CNOT kanalima, a Bobove potencijale Bornovim verovatnoćama Qiskit dekodera. 
Piramidu računam preko malih matrica gustine, bez gustog stanja sa 39 kubita. 
Mešavina dve izlazne raspodele.
"""
