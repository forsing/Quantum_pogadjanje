#!/usr/bin/env python3

# Quantum_pogadjanje


"""
QUANTUM POGADJANJE V1 — Qiskit EARAC — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Quantum predictive path:
    historical occupancy triples;
    learned Alice measurement directions;
    Qiskit two-qubit singlet;
    Alice measurement and one-bit communication;
    Bob's X/Y/Z query;
    Born probabilities of the decoded bit.

The learned parameters are Alice measurement-angle offsets.
There is no learned classical readout regressor.

Each query denotes a separate preparation and readout.
Incompatible observables are not measured on the same copy.

For lookback > 1, decoded probabilities are averaged over the
available preceding draws. This is an exact mixture of separate
protocol preparations, without random sampling.

Fresh preparations define independent candidate-bit outcomes.
The output distribution is conditioned on exactly seven ones:

    h_i = log(q_i) - log(1-q_i)

    P(S|history) =
        exp(sum(h_i for i in S)) / e_7(exp(h))

The normalizer and inclusion marginals are computed exactly.
The maximum is the seven largest h_i values, with deterministic
lexicographic tie handling. A small exhaustive check verifies it.

This is a temporal adaptation of EARAC.
The original protocol reconstructs a known input bit.
Learning here uses preceding histories to fit future draws.

No random initialization, sampling, injected noise, shuffling,
classifier, rounding, duplicate repair or cosmetic filters.
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
from itertools import combinations
from pathlib import Path

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)

NUMBER_COUNT = 39
DRAW_SIZE = 7
GROUP_COUNT = 13

CANDIDATES = (
    (1, 0.03),
    (1, 0.30),
    (3, 0.03),
    (3, 0.30),
)

MAXITER = 250
CONTINUATIONS = 2
GRADIENT_LIMIT = 2e-4


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class QuantumEARAC:
    """
    Qubit 0: Alice.
    Qubit 1: Bob.

    Qiskit Pauli strings are little-endian:
        XZ = Bob X tensor Alice Z
        YZ = Bob Y tensor Alice Z
        ZZ = Bob Z tensor Alice Z

    Alice sends:
        message = first_history_bit XOR alice_outcome

    Bob decodes:
        guess = message XOR (1 - bob_outcome)

    Therefore:
        P(guess=1) = (1 + s0 * <AliceZ BobAxis>) / 2
    where s0 = (-1)^first_history_bit.
    """

    def __init__(self, groups):
        self.groups = int(groups)
        self.m = 3 * self.groups
        self.parameter_count = self.groups * 8 * 2

        require(self.groups >= 1, "Neispravan broj trobitnih grupa.")

        self.bits = np.array(
            [
                [(code >> 2) & 1, (code >> 1) & 1, code & 1]
                for code in range(8)
            ],
            dtype=np.int64,
        )

        self.signs = 1.0 - 2.0 * self.bits
        self.first_sign = self.signs[:, 0]

        axes = (
            self.signs * self.first_sign[:, None]
            / math.sqrt(3.0)
        )

        self.base_polar = np.arccos(axes[:, 2])
        self.base_azimuth = np.arctan2(
            axes[:, 1],
            axes[:, 0],
        )

        self.observables = (
            Pauli("XZ"),
            Pauli("YZ"),
            Pauli("ZZ"),
        )

        circuit = QuantumCircuit(2, name="Singlet")
        circuit.x(0)
        circuit.h(1)
        circuit.cx(1, 0)
        circuit.z(1)

        self.singlet_circuit = circuit
        self.singlet = Statevector.from_instruction(circuit)

        require(
            abs(float(np.vdot(
                self.singlet.data,
                self.singlet.data,
            ).real) - 1.0) < 1e-12,
            "Singlet nije normalizovan.",
        )

    def angles(self, parameters):
        parameters = np.asarray(parameters, dtype=float)

        require(
            parameters.shape == (self.parameter_count,)
            and np.all(np.isfinite(parameters)),
            "Neispravni kvantni parametri.",
        )

        offsets = parameters.reshape(self.groups, 8, 2)

        polar = self.base_polar[None, :] + offsets[:, :, 0]
        azimuth = self.base_azimuth[None, :] + offsets[:, :, 1]

        return polar, azimuth

    def alice_state(self, polar, azimuth):
        circuit = QuantumCircuit(2, name="AliceMeasurementBasis")
        circuit.rz(-float(azimuth), 0)
        circuit.ry(-float(polar), 0)

        return self.singlet.evolve(circuit)

    def lookup(self, parameters, with_gradient=False):
        polar, azimuth = self.angles(parameters)

        probabilities = np.empty(
            (self.groups, 8, 3),
            dtype=float,
        )

        for group in range(self.groups):
            for code in range(8):
                state = self.alice_state(
                    polar[group, code],
                    azimuth[group, code],
                )

                require(
                    np.all(np.isfinite(state.data)),
                    "Qiskit je vratio nevažeće amplitude.",
                )

                require(
                    abs(float(np.vdot(
                        state.data,
                        state.data,
                    ).real) - 1.0) < 2e-10,
                    "Qiskit stanje nije normalizovano.",
                )

                for axis, observable in enumerate(self.observables):
                    correlation = complex(
                        state.expectation_value(observable)
                    )

                    require(
                        abs(correlation.imag) < 1e-10,
                        "Korelacija ima imaginarni ostatak.",
                    )

                    probabilities[group, code, axis] = (
                        1.0
                        + self.first_sign[code] * correlation.real
                    ) / 2.0

        # Do not silently clip invalid or zero probabilities.
        require(
            np.all(np.isfinite(probabilities))
            and np.all(probabilities > 0.0)
            and np.all(probabilities < 1.0),
            "EARAC izlaz nije strogo između nule i jedan.",
        )

        if not with_gradient:
            return probabilities, None

        sine_polar = np.sin(polar)
        cosine_polar = np.cos(polar)
        sine_azimuth = np.sin(azimuth)
        cosine_azimuth = np.cos(azimuth)

        axis = np.stack(
            (
                sine_polar * cosine_azimuth,
                sine_polar * sine_azimuth,
                cosine_polar,
            ),
            axis=-1,
        )

        reference = (
            1.0
            - self.first_sign[None, :, None] * axis
        ) / 2.0

        require(
            np.max(np.abs(
                probabilities - reference
            )) < 2e-10,
            "Qiskit Born izlaz nije saglasan sa singlet protokolom.",
        )

        derivative_polar = np.stack(
            (
                cosine_polar * cosine_azimuth,
                cosine_polar * sine_azimuth,
                -sine_polar,
            ),
            axis=-1,
        )

        derivative_azimuth = np.stack(
            (
                -sine_polar * sine_azimuth,
                sine_polar * cosine_azimuth,
                np.zeros_like(polar),
            ),
            axis=-1,
        )

        factor = -0.5 * self.first_sign[None, :, None]

        derivatives = np.stack(
            (
                factor * derivative_polar,
                factor * derivative_azimuth,
            ),
            axis=-1,
        )

        return probabilities, derivatives


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    blank = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not value.strip() for value in row):
                blank += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = tuple(int(value.strip()) for value in row)
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

    require(
        len(rows) >= 100,
        f"{path.name}: nedovoljno istorije.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=float,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    triples = occupancy.astype(np.int64).reshape(
        len(draws),
        GROUP_COUNT,
        3,
    )

    codes = (
        4 * triples[:, :, 0]
        + 2 * triples[:, :, 1]
        + triples[:, :, 2]
    )

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, codes, fingerprint


def make_context(codes, targets, lookback):
    targets = np.asarray(targets, dtype=np.int64)

    require(
        targets.ndim == 1
        and np.all(targets >= 1)
        and np.all(targets <= len(codes)),
        "Neispravni indeksi istorijskih konteksta.",
    )

    result = np.full(
        (len(targets), lookback, codes.shape[1]),
        -1,
        dtype=np.int64,
    )

    for lag in range(1, lookback + 1):
        previous = targets - lag
        valid = previous >= 0
        result[valid, lag - 1, :] = codes[previous[valid]]

    return result


def contextual_probabilities(lookup, context):
    n, lookback, groups = context.shape

    require(
        lookup.shape == (groups, 8, 3),
        "Pogrešan oblik kvantnog očitavanja.",
    )

    result = np.zeros((n, groups, 3), dtype=float)
    count = np.zeros(n, dtype=float)
    group_indices = np.arange(groups)[None, :]

    for lag in range(lookback):
        valid = context[:, lag, 0] >= 0

        if np.any(valid):
            selected = context[valid, lag, :]

            require(
                np.all((selected >= 0) & (selected < 8)),
                "Neispravan istorijski trobitni kod.",
            )

            result[valid] += lookup[
                group_indices,
                selected,
            ]

            count[valid] += 1.0

    require(
        np.all(count > 0),
        "Istorijski kontekst nema raspoloživo prethodno kolo.",
    )

    result /= count[:, None, None]
    probabilities = result.reshape(n, groups * 3)

    require(
        np.all(probabilities > 0)
        and np.all(probabilities < 1)
        and np.all(np.isfinite(probabilities)),
        "Neispravna istorijski uslovljena Born raspodela.",
    )

    return probabilities, count


def fixed_size_partition(fields, k, with_marginals=False):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 2
        and np.all(np.isfinite(fields))
        and 1 <= k <= fields.shape[1],
        "Neispravan ulaz normalizacije sedmorke.",
    )

    n, m = fields.shape

    forward = np.full(
        (n, m + 1, k + 1),
        -np.inf,
    )

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

    marginals = np.empty_like(fields)

    for number in range(m):
        terms = (
            forward[:, number, :k]
            + backward[:, number + 1, k - 1::-1]
        )

        marginals[:, number] = np.exp(
            fields[:, number]
            + logsumexp(terms, axis=1)
            - log_z
        )

    require(
        np.all(np.isfinite(marginals))
        and np.min(marginals) >= -1e-9
        and np.max(marginals) <= 1.0 + 1e-9,
        "Uslovne marginalne verovatnoće nisu ispravne.",
    )

    require(
        np.max(np.abs(
            marginals.sum(axis=1) - k
        )) < 1e-7,
        "Zbir uslovnih marginalnih verovatnoća nije sedam.",
    )

    return log_z, marginals


def fields_from_probabilities(probabilities):
    require(
        np.all(probabilities > 0)
        and np.all(probabilities < 1),
        "Born verovatnoće nisu u otvorenom intervalu (0,1).",
    )

    return (
        np.log(probabilities)
        - np.log1p(-probabilities)
    )


def objective(
    parameters,
    source,
    context,
    targets,
    ridge,
    k,
    gradient=True,
):
    lookup, derivatives = source.lookup(
        parameters,
        with_gradient=gradient,
    )

    probabilities, count = contextual_probabilities(
        lookup,
        context,
    )

    fields = fields_from_probabilities(probabilities)

    log_z, marginals = fixed_size_partition(
        fields,
        k,
        with_marginals=gradient,
    )

    target_scores = np.sum(fields * targets, axis=1)
    logp = target_scores - log_z

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća ciljne kombinacije.",
    )

    penalty = ridge * float(np.mean(parameters * parameters))
    loss = -float(logp.mean()) + penalty

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    n = len(targets)
    groups = source.groups

    probability_gradient = (
        (marginals - targets)
        / (probabilities * (1.0 - probabilities))
        / n
    ).reshape(n, groups, 3)

    lookup_gradient = np.zeros_like(lookup)

    for lag in range(context.shape[1]):
        valid_indices = np.flatnonzero(
            context[:, lag, 0] >= 0
        )

        if not len(valid_indices):
            continue

        for group in range(groups):
            codes = context[valid_indices, lag, group]

            contribution = (
                probability_gradient[valid_indices, group, :]
                / count[valid_indices, None]
            )

            np.add.at(
                lookup_gradient[group],
                codes,
                contribution,
            )

    result = np.sum(
        lookup_gradient[:, :, :, None] * derivatives,
        axis=2,
    ).ravel()

    result += (
        2.0 * ridge * parameters / len(parameters)
    )

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


@dataclass
class Model:
    source: QuantumEARAC
    parameters: np.ndarray
    lookup: np.ndarray
    lookback: int
    ridge: float
    loss: float
    gradient_norm: float

    def predict(self, context, k, marginals=False):
        probabilities, _ = contextual_probabilities(
            self.lookup,
            context,
        )

        fields = fields_from_probabilities(probabilities)

        log_z, inclusion = fixed_size_partition(
            fields,
            k,
            with_marginals=marginals,
        )

        return probabilities, fields, log_z, inclusion


def fit_model(context, targets, lookback, ridge, label):
    source = QuantumEARAC(context.shape[2])

    # The deterministic zero offsets reproduce the original
    # three-bit EARAC directions before training.
    parameters = np.zeros(source.parameter_count, dtype=float)

    started = time.perf_counter()
    iteration = 0

    say(
        "",
        label,
        f"Lag={lookback} | λ={ridge:g}"
        f" | prelaza={len(targets)}",
        f"Kvantnih ugaonih parametara: {source.parameter_count}",
    )

    def function(current):
        return objective(
            current,
            source,
            context,
            targets,
            ridge,
            DRAW_SIZE,
        )

    def callback(current):
        nonlocal iteration
        iteration += 1

        if iteration % 25 == 0:
            elapsed = time.perf_counter() - started

            say(
                f"Iteracija {iteration}"
                f" | proteklo {elapsed:.1f} s"
            )

    converged = False
    result = None

    for continuation in range(CONTINUATIONS):
        result = minimize(
            function,
            parameters,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": MAXITER,
                "maxls": 40,
                "maxcor": 15,
                "ftol": 1e-12,
                "gtol": 1e-6,
            },
        )

        parameters = result.x.copy()
        loss, gradient = function(parameters)
        maximum_gradient = float(np.max(np.abs(gradient)))

        if result.success and maximum_gradient <= GRADIENT_LIMIT:
            converged = True
            break

        say(
            f"Nastavak {continuation + 1}",
            str(result.message),
            f"Max|grad|: {maximum_gradient:.3e}",
        )

    require(
        converged,
        f"{label}: optimizacija nije prošla uslov konvergencije.",
    )

    lookup, _ = source.lookup(parameters)

    say(
        f"Prihvaćen lokalni minimum: {loss:.9f}",
        f"Max|grad|: {maximum_gradient:.3e}",
        f"Trajanje: {time.perf_counter() - started:.1f} s",
    )

    return Model(
        source=source,
        parameters=parameters.copy(),
        lookup=lookup.copy(),
        lookback=lookback,
        ridge=ridge,
        loss=float(loss),
        gradient_norm=maximum_gradient,
    )


def evaluate_nll(model, context, targets):
    _, fields, log_z, _ = model.predict(
        context,
        DRAW_SIZE,
    )

    logp = np.sum(fields * targets, axis=1) - log_z

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna evaluaciona verovatnoća.",
    )

    return -float(logp.mean())


def exact_maximum(fields, k):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 1
        and np.all(np.isfinite(fields))
        and 1 <= k <= len(fields),
        "Neispravna polja za izbor maksimuma.",
    )

    # Maximize the additive set log-probability exactly.
    # Equal fields prefer the smaller label.
    order = sorted(
        range(len(fields)),
        key=lambda label: (-float(fields[label]), label),
    )

    return tuple(sorted(order[:k]))


def verify_explicit_messages(source):
    expected_success = (
        1.0 + 1.0 / math.sqrt(3.0)
    ) / 2.0

    lookup, _ = source.lookup(
        np.zeros(source.parameter_count)
    )

    for code in range(8):
        polar = source.base_polar[code]
        azimuth = source.base_azimuth[code]
        alice_state = source.alice_state(polar, azimuth)

        for axis in range(3):
            readout = QuantumCircuit(2)

            if axis == 0:
                readout.h(1)
            elif axis == 1:
                readout.sdg(1)
                readout.h(1)

            final_state = alice_state.evolve(readout)
            probabilities = np.abs(final_state.data) ** 2

            decoded_one = 0.0
            success = 0.0

            for index, probability in enumerate(probabilities):
                alice_outcome = index & 1
                bob_outcome = (index >> 1) & 1

                message = (
                    int(source.bits[code, 0]) ^ alice_outcome
                )

                guess = message ^ (1 - bob_outcome)

                if guess == 1:
                    decoded_one += float(probability)

                if guess == int(source.bits[code, axis]):
                    success += float(probability)

            require(
                abs(float(probabilities.sum()) - 1.0) < 1e-12,
                "Zajedničke Born verovatnoće nisu normalizovane.",
            )

            require(
                abs(
                    decoded_one - lookup[0, code, axis]
                ) < 1e-12,
                "Eksplicitna Alice–Bob poruka nije saglasna sa izlazom.",
            )

            require(
                abs(success - expected_success) < 1e-12,
                "Originalni trobitni EARAC protokol nije prošao.",
            )


def self_checks():
    say(
        "",
        "Pokrećem kratke ugrađene kvantne i matematičke provere.",
    )

    started = time.perf_counter()

    source = QuantumEARAC(2)
    verify_explicit_messages(source)

    parameters = 0.04 * np.sin(
        np.arange(source.parameter_count) + 1.0
    )

    context = np.array(
        [
            [[0, 1], [-1, -1]],
            [[2, 4], [1, 0]],
            [[5, 3], [2, 4]],
            [[7, 6], [5, 3]],
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )

    k = 2
    ridge = 0.03

    _, analytic = objective(
        parameters,
        source,
        context,
        targets,
        ridge,
        k,
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
                positive,
                source,
                context,
                targets,
                ridge,
                k,
                gradient=False,
            )
            - objective(
                negative,
                source,
                context,
                targets,
                ridge,
                k,
                gradient=False,
            )
        ) / (2.0 * epsilon)

    gradient_error = float(
        np.max(np.abs(analytic - numerical))
    )

    require(
        gradient_error < 3e-6,
        f"Provera ukupnog gradijenta nije prošla: {gradient_error:.3e}",
    )

    lookup, _ = source.lookup(parameters)
    probabilities, _ = contextual_probabilities(lookup, context)
    fields = fields_from_probabilities(probabilities)

    log_z, marginals = fixed_size_partition(
        fields,
        k,
        with_marginals=True,
    )

    subsets = list(combinations(range(source.m), k))
    selected = np.asarray(subsets, dtype=np.int64)

    for sample in range(len(context)):
        scores = fields[sample][selected].sum(axis=1)
        direct_log_z = float(logsumexp(scores))
        direct_probabilities = np.exp(scores - direct_log_z)

        require(
            abs(direct_log_z - log_z[sample]) < 1e-11,
            "Potpuna provera normalizacije nije prošla.",
        )

        require(
            abs(float(direct_probabilities.sum()) - 1.0) < 1e-12,
            "Zbir direktnih verovatnoća nije jedan.",
        )

        direct_marginals = np.zeros(source.m)

        for subset, probability in zip(
            subsets,
            direct_probabilities,
        ):
            direct_marginals[np.asarray(subset)] += probability

        require(
            np.max(np.abs(
                direct_marginals - marginals[sample]
            )) < 1e-11,
            "Potpuna provera margina nije prošla.",
        )

        maximum = exact_maximum(fields[sample], k)
        direct_maximum = subsets[int(np.argmax(scores))]

        require(
            maximum == direct_maximum,
            "Tačan izbor maksimuma nije saglasan sa enumeracijom.",
        )

    for boundary_k in (1, source.m):
        boundary_z, boundary_marginals = fixed_size_partition(
            np.zeros((1, source.m)),
            boundary_k,
            with_marginals=True,
        )

        require(
            abs(
                boundary_z[0]
                - math.log(math.comb(source.m, boundary_k))
            ) < 1e-11,
            "Granična normalizacija nije prošla.",
        )

        require(
            np.max(np.abs(
                boundary_marginals[0]
                - boundary_k / source.m
            )) < 1e-11,
            "Granične margine nisu prošle.",
        )

    zero_tie = exact_maximum(np.zeros(6), 2)

    require(
        zero_tie == (0, 1),
        "Deterministička obrada jednakih maksimuma nije prošla.",
    )

    toy_codes = np.array(
        [[0, 1], [2, 3], [4, 5]],
        dtype=np.int64,
    )

    first_context = make_context(
        toy_codes,
        [1],
        lookback=3,
    )

    require(
        np.array_equal(first_context[0, 0], toy_codes[0])
        and np.all(first_context[0, 1:] == -1),
        "Prvi istorijski prelaz nije ispravan.",
    )

    say(
        "Ugrađene provere su prošle.",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
        "Provereni su Qiskit protokol, poruka, normalizacija, margine i maksimum.",
        f"Trajanje provera: {time.perf_counter() - started:.1f} s",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def process_file(path, draws, occupancy, codes, fingerprint):
    targets = occupancy[1:].copy()
    n = len(targets)

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno podataka za hronološku podelu.",
    )

    contexts = {
        lookback: make_context(
            codes,
            np.arange(1, len(codes)),
            lookback,
        )
        for lookback in sorted(
            {candidate[0] for candidate in CANDIDATES}
        )
    }

    folds = (
        (split60, split70),
        (split70, split80),
    )

    scores = []

    for lookback, ridge in CANDIDATES:
        context = contexts[lookback]
        total_nll = 0.0
        total_count = 0

        for fold, (train_end, validation_end) in enumerate(folds, 1):
            model = fit_model(
                context[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                f"{path.name} / validacija {fold}",
            )

            count = validation_end - train_end

            score = evaluate_nll(
                model,
                context[train_end:validation_end],
                targets[train_end:validation_end],
            )

            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count
        scores.append((score, lookback, ridge))

        say(
            f"KANDIDAT: lag={lookback} | λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
        )

    validation_nll, lookback, ridge = min(
        scores,
        key=lambda item: item[0],
    )

    context = contexts[lookback]

    say(
        "",
        f"IZABRANO: lag={lookback} | λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        context[:split80],
        targets[:split80],
        lookback,
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        context[split80:],
        targets[split80:],
    )

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"NLL cele sedmorke: {test_nll:.9f}",
    )

    final_model = fit_model(
        context,
        targets,
        lookback,
        ridge,
        f"{path.name} / završno učenje",
    )

    next_context = make_context(
        codes,
        [len(codes)],
        lookback,
    )

    # Re-execute the final learned Qiskit protocol.
    final_model.lookup, _ = final_model.source.lookup(
        final_model.parameters
    )

    q_next, fields, log_z, inclusion = final_model.predict(
        next_context,
        DRAW_SIZE,
        marginals=True,
    )

    best = exact_maximum(fields[0], DRAW_SIZE)
    logp = float(
        np.sum(fields[0, np.asarray(best)]) - log_z[0]
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

    previous_q, _, _, previous_inclusion = final_model.predict(
        context[-8:],
        DRAW_SIZE,
        marginals=True,
    )

    q_sensitivity = float(np.max(np.abs(
        previous_q - q_next[0]
    )))

    marginal_sensitivity = float(np.max(np.abs(
        previous_inclusion - inclusion[0]
    )))

    # Report the original protocol's initial maximum separately.
    # This tests whether fitting changed the reconstruction prior.
    initial_lookup, _ = final_model.source.lookup(
        np.zeros(final_model.source.parameter_count)
    )

    initial_q, _ = contextual_probabilities(
        initial_lookup,
        next_context,
    )

    initial_best = exact_maximum(
        fields_from_probabilities(initial_q)[0],
        DRAW_SIZE,
    )

    warnings = []

    run = longest_run(best)
    span = best[-1] - best[0]

    repetitions = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if span <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva je u rasponu od deset brojeva."
        )

    if repetitions:
        warnings.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji."
        )

    if best == tuple(int(value) for value in draws[-1]):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if best == initial_best:
        warnings.append(
            "Naučeni NEXT maksimum je isti kao početni EARAC maksimum."
        )

    if marginal_sensitivity < 1e-8:
        warnings.append(
            "Uslovne margine slabo reaguju na proverene istorijske kontekste."
        )

    say(
        "",
        f"ZAVRŠNO UČENJE: svih {n} prelaza",
        "Početni EARAC maksimum: "
        + " ".join(f"{value + 1:02d}" for value in initial_best),
        "Naučeni NEXT: "
        + " ".join(f"{value + 1:02d}" for value in best),
        f"Promena Born verovatnoća zbog istorije: {q_sensitivity:.9e}",
        f"Promena uslovnih margina zbog istorije: {marginal_sensitivity:.9e}",
        f"Zbir uslovnih margina: {float(inclusion.sum()):.12f}",
    )

    return {
        "name": path.name,
        "hash": fingerprint,
        "next": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "probability": math.exp(logp),
        "marginal_sum": float(inclusion.sum()),
        "q_sensitivity": q_sensitivity,
        "marginal_sensitivity": marginal_sensitivity,
        "warnings": warnings,
    }


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    require(
        NUMBER_COUNT == 3 * GROUP_COUNT,
        "Broj oznaka nije usaglašen sa trobitnim grupama.",
    )

    say(
        "QUANTUM POGADJANJE V1 — Qiskit EARAC — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Alice–Bob singlet protokol koristi dva kubita po pripremi.",
        "Istorija ima trinaest trobitnih grupa.",
        "Uče se uglovi Aliceinih kvantnih merenja.",
        "Izlaz dolazi iz Qiskit amplituda i Bornovih verovatnoća.",
        "Svaki Bobov upit koristi zasebnu pripremu stanja.",
        "Nema klasičnog prediktivnog regresora.",
        "Klasični optimizator podešava kvantne uglove.",
        "Raspodela se tačno uslovljava na sedam različitih brojeva.",
        "Uslovljavanje i normalizacioni proračun su klasični.",
        "Nema slučajnog uzorkovanja, šuma ili mešanja redova.",
        "Maksimum se određuje tačno bez pretrage 15 miliona sedmorki.",
        "Uslovljena raspodela nema dodatne naučene interakcije među izlaznim brojevima.",
        "Optimizacija je lokalna; globalni minimum nije garantovan.",
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

        draws, occupancy, codes, fingerprint = load_draws(path)

        datasets.append(
            (path, draws, occupancy, codes, fingerprint)
        )

        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    # Full training starts only after these checks pass.
    self_checks()

    results = [
        process_file(path, draws, occupancy, codes, fingerprint)
        for path, draws, occupancy, codes, fingerprint in datasets
    ]

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
        numbers = " ".join(
            f"{value:02d}" for value in result["next"]
        )

        say(
            "",
            result["name"],
            f"NEXT: {numbers}",
            f"Validacioni NLL: {result['validation_nll']:.9f}",
            f"Odvojeni test NLL: {result['test_nll']:.9f}",
            f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
            f"Zbir uslovnih margina: {result['marginal_sum']:.12f}",
            "Osetljivost uslovnih margina:"
            f" {result['marginal_sensitivity']:.9e}",
        )

        for warning in result["warnings"]:
            say(f"DIJAGNOSTIKA: {warning}")

    say(
        "",
        "Obrada je završena.",
        "NEXT je tačan maksimum raspodele za naučene parametre.",
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
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)
    


"""
QUANTUM POGADJANJE V1 — Qiskit EARAC — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Alice-Bob singlet protokol koristi dva kubita po pripremi.
Istorija ima trinaest trobitnih grupa.
Uče se uglovi Aliceinih kvantnih merenja.
Izlaz dolazi iz Qiskit amplituda i Bornovih verovatnoća.
Svaki Bobov upit koristi zasebnu pripremu stanja.
Nema klasičnog prediktivnog regresora.
Klasični optimizator podešava kvantne uglove.
Raspodela se tačno uslovljava na sedam različitih brojeva.
Uslovljavanje i normalizacioni proračun su klasični.
Nema slučajnog uzorkovanja, šuma ili mešanja redova.
Maksimum se određuje tačno bez pretrage 15 miliona sedmorki.
Uslovljena raspodela nema dodatne naučene interakcije među izlaznim brojevima.
Optimizacija je lokalna; globalni minimum nije garantovan.

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

Pokrećem kratke ugrađene kvantne i matematičke provere.
Ugrađene provere su prošle.
Najveća greška gradijenta: 4.751e-10
Provereni su Qiskit protokol, poruka, normalizacija, margine i maksimum.
Trajanje provera: 0.2 s

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.03 | prelaza=2825
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.8 s
Iteracija 50 | proteklo 3.1 s
Iteracija 75 | proteklo 4.6 s
Iteracija 100 | proteklo 6.0 s
Iteracija 125 | proteklo 7.3 s
Iteracija 150 | proteklo 8.9 s
Iteracija 175 | proteklo 10.2 s
Iteracija 200 | proteklo 11.6 s
Prihvaćen lokalni minimum: 16.663855641
Max|grad|: 9.387e-07
Trajanje: 12.4 s
Validacioni NLL: 16.671135486

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.03 | prelaza=3296
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.7 s
Iteracija 50 | proteklo 3.5 s
Iteracija 75 | proteklo 5.0 s
Iteracija 100 | proteklo 6.5 s
Iteracija 125 | proteklo 8.0 s
Prihvaćen lokalni minimum: 16.674999558
Max|grad|: 7.887e-07
Trajanje: 9.3 s
Validacioni NLL: 16.795434975
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.733285231

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.3 | prelaza=2825
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.8 s
Iteracija 50 | proteklo 3.2 s
Iteracija 75 | proteklo 4.6 s
Iteracija 100 | proteklo 6.0 s
Iteracija 125 | proteklo 7.7 s
Iteracija 150 | proteklo 9.1 s
Prihvaćen lokalni minimum: 17.031508753
Max|grad|: 9.461e-07
Trajanje: 10.4 s
Validacioni NLL: 16.784244664

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.3 | prelaza=3296
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.8 s
Iteracija 50 | proteklo 3.3 s
Iteracija 75 | proteklo 4.9 s
Iteracija 100 | proteklo 6.6 s
Iteracija 125 | proteklo 8.2 s
Iteracija 150 | proteklo 9.7 s
Prihvaćen lokalni minimum: 17.039779178
Max|grad|: 8.776e-07
Trajanje: 10.3 s
Validacioni NLL: 16.843559632
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 16.813902148

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.03 | prelaza=2825
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.5 s
Iteracija 50 | proteklo 3.1 s
Iteracija 75 | proteklo 4.5 s
Iteracija 100 | proteklo 6.0 s
Iteracija 125 | proteklo 7.5 s
Iteracija 150 | proteklo 8.9 s
Iteracija 175 | proteklo 10.3 s
Iteracija 200 | proteklo 11.8 s
Iteracija 225 | proteklo 13.3 s
Prihvaćen lokalni minimum: 16.560642466
Max|grad|: 8.667e-07
Trajanje: 14.0 s
Validacioni NLL: 16.586298168

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.03 | prelaza=3296
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.7 s
Iteracija 50 | proteklo 3.3 s
Iteracija 75 | proteklo 4.9 s
Iteracija 100 | proteklo 6.6 s
Iteracija 125 | proteklo 8.1 s
Iteracija 150 | proteklo 9.7 s
Iteracija 175 | proteklo 11.4 s
Iteracija 200 | proteklo 13.0 s
Iteracija 225 | proteklo 14.6 s
Iteracija 250 | proteklo 16.3 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 3.197e-06
Prihvaćen lokalni minimum: 16.566249158
Max|grad|: 8.713e-07
Trajanje: 16.9 s
Validacioni NLL: 16.583414545
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.584856356

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.3 | prelaza=2825
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.7 s
Iteracija 50 | proteklo 3.1 s
Iteracija 75 | proteklo 4.7 s
Iteracija 100 | proteklo 6.3 s
Prihvaćen lokalni minimum: 16.829101132
Max|grad|: 7.264e-07
Trajanje: 7.2 s
Validacioni NLL: 16.653176740

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.3 | prelaza=3296
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.8 s
Iteracija 50 | proteklo 3.4 s
Iteracija 75 | proteklo 5.0 s
Iteracija 100 | proteklo 6.7 s
Prihvaćen lokalni minimum: 16.834557007
Max|grad|: 9.359e-07
Trajanje: 7.1 s
Validacioni NLL: 16.661436975
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.657306858

IZABRANO: lag=3 | λ=0.03
Validacioni NLL: 16.584856356

loto7_4710_k86.csv / odvojeni test
Lag=3 | λ=0.03 | prelaza=3767
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 2.1 s
Iteracija 50 | proteklo 3.9 s
Iteracija 75 | proteklo 5.7 s
Iteracija 100 | proteklo 7.5 s
Iteracija 125 | proteklo 9.3 s
Iteracija 150 | proteklo 11.1 s
Iteracija 175 | proteklo 12.9 s
Iteracija 200 | proteklo 14.7 s
Iteracija 225 | proteklo 16.4 s
Iteracija 250 | proteklo 18.2 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 2.433e-06
Iteracija 275 | proteklo 20.3 s
Prihvaćen lokalni minimum: 16.569588663
Max|grad|: 9.309e-07
Trajanje: 20.4 s
ODVOJENI TEST: 942 prelaza
NLL cele sedmorke: 16.571771674

loto7_4710_k86.csv / završno učenje
Lag=3 | λ=0.03 | prelaza=4709
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 2.4 s
Iteracija 50 | proteklo 4.5 s
Iteracija 75 | proteklo 6.7 s
Iteracija 100 | proteklo 9.0 s
Iteracija 125 | proteklo 11.2 s
Iteracija 150 | proteklo 13.4 s
Iteracija 175 | proteklo 15.6 s
Iteracija 200 | proteklo 17.9 s
Iteracija 225 | proteklo 20.1 s
Iteracija 250 | proteklo 22.4 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 3.227e-06
Prihvaćen lokalni minimum: 16.573475226
Max|grad|: 8.189e-07
Trajanje: 23.2 s

ZAVRŠNO UČENJE: svih 4709 prelaza
Početni EARAC maksimum: 03 06 09 10 13 26 32
Naučeni NEXT: 08 13 16 21 26 34 39
Promena Born verovatnoća zbog istorije: 6.849195044e-02
Promena uslovnih margina zbog istorije: 6.247465902e-02
Zbir uslovnih margina: 7.000000000000

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.03 | prelaza=1785
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.1 s
Iteracija 50 | proteklo 2.2 s
Iteracija 75 | proteklo 3.2 s
Iteracija 100 | proteklo 4.3 s
Iteracija 125 | proteklo 5.3 s
Iteracija 150 | proteklo 6.3 s
Iteracija 175 | proteklo 7.3 s
Iteracija 200 | proteklo 8.3 s
Iteracija 225 | proteklo 9.3 s
Prihvaćen lokalni minimum: 16.593988524
Max|grad|: 6.962e-07
Trajanje: 9.8 s
Validacioni NLL: 16.713192360

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.03 | prelaza=2083
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.2 s
Iteracija 50 | proteklo 2.3 s
Iteracija 75 | proteklo 3.4 s
Iteracija 100 | proteklo 4.6 s
Iteracija 125 | proteklo 5.7 s
Iteracija 150 | proteklo 6.7 s
Iteracija 175 | proteklo 7.8 s
Iteracija 200 | proteklo 9.0 s
Iteracija 225 | proteklo 10.1 s
Iteracija 250 | proteklo 11.2 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 1.779e-06
Prihvaćen lokalni minimum: 16.584648267
Max|grad|: 9.401e-07
Trajanje: 11.5 s
Validacioni NLL: 16.648922106
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.681111242

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.3 | prelaza=1785
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.1 s
Iteracija 50 | proteklo 2.1 s
Iteracija 75 | proteklo 3.2 s
Iteracija 100 | proteklo 4.2 s
Prihvaćen lokalni minimum: 16.959067745
Max|grad|: 9.022e-07
Trajanje: 5.1 s
Validacioni NLL: 16.781374085

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.3 | prelaza=2083
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.3 s
Iteracija 50 | proteklo 2.4 s
Iteracija 75 | proteklo 3.5 s
Iteracija 100 | proteklo 4.7 s
Iteracija 125 | proteklo 5.7 s
Prihvaćen lokalni minimum: 16.959725938
Max|grad|: 9.581e-07
Trajanje: 6.9 s
Validacioni NLL: 16.743878410
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 16.762657756

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.03 | prelaza=1785
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.1 s
Iteracija 50 | proteklo 2.2 s
Iteracija 75 | proteklo 3.3 s
Iteracija 100 | proteklo 4.5 s
Iteracija 125 | proteklo 5.6 s
Iteracija 150 | proteklo 6.6 s
Iteracija 175 | proteklo 7.8 s
Iteracija 200 | proteklo 8.8 s
Iteracija 225 | proteklo 9.9 s
Prihvaćen lokalni minimum: 16.532322337
Max|grad|: 8.786e-07
Trajanje: 10.9 s
Validacioni NLL: 16.596856940

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.03 | prelaza=2083
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.3 s
Iteracija 50 | proteklo 2.5 s
Iteracija 75 | proteklo 3.7 s
Iteracija 100 | proteklo 4.9 s
Iteracija 125 | proteklo 6.1 s
Iteracija 150 | proteklo 7.3 s
Iteracija 175 | proteklo 8.5 s
Iteracija 200 | proteklo 9.8 s
Prihvaćen lokalni minimum: 16.539270027
Max|grad|: 6.403e-07
Trajanje: 10.4 s
Validacioni NLL: 16.607396414
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.602117820

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.3 | prelaza=1785
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.2 s
Iteracija 50 | proteklo 2.3 s
Iteracija 75 | proteklo 3.4 s
Iteracija 100 | proteklo 4.5 s
Iteracija 125 | proteklo 5.6 s
Prihvaćen lokalni minimum: 16.811248275
Max|grad|: 4.175e-07
Trajanje: 6.2 s
Validacioni NLL: 16.680466996

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.3 | prelaza=2083
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.3 s
Iteracija 50 | proteklo 2.6 s
Iteracija 75 | proteklo 3.8 s
Iteracija 100 | proteklo 5.1 s
Iteracija 125 | proteklo 6.3 s
Prihvaćen lokalni minimum: 16.819849269
Max|grad|: 9.571e-07
Trajanje: 6.5 s
Validacioni NLL: 16.663410163
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.671952913

IZABRANO: lag=3 | λ=0.03
Validacioni NLL: 16.602117820

loto7_4710_k86_loto_2977.csv / odvojeni test
Lag=3 | λ=0.03 | prelaza=2380
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.5 s
Iteracija 50 | proteklo 2.8 s
Iteracija 75 | proteklo 4.2 s
Iteracija 100 | proteklo 5.5 s
Iteracija 125 | proteklo 6.8 s
Iteracija 150 | proteklo 8.1 s
Iteracija 175 | proteklo 9.4 s
Iteracija 200 | proteklo 10.8 s
Iteracija 225 | proteklo 12.1 s
Prihvaćen lokalni minimum: 16.546959519
Max|grad|: 7.745e-07
Trajanje: 12.7 s
ODVOJENI TEST: 596 prelaza
NLL cele sedmorke: 16.594303055

loto7_4710_k86_loto_2977.csv / završno učenje
Lag=3 | λ=0.03 | prelaza=2976
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 2.0 s
Iteracija 50 | proteklo 3.5 s
Iteracija 75 | proteklo 5.0 s
Iteracija 100 | proteklo 6.6 s
Iteracija 125 | proteklo 8.1 s
Iteracija 150 | proteklo 9.5 s
Iteracija 175 | proteklo 11.0 s
Iteracija 200 | proteklo 12.5 s
Iteracija 225 | proteklo 14.0 s
Iteracija 250 | proteklo 15.6 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 8.448e-06
Iteracija 275 | proteklo 17.4 s
Iteracija 300 | proteklo 18.9 s
Iteracija 325 | proteklo 20.5 s
Prihvaćen lokalni minimum: 16.554624981
Max|grad|: 8.951e-07
Trajanje: 21.1 s

ZAVRŠNO UČENJE: svih 2976 prelaza
Početni EARAC maksimum: 01 03 06 09 15 22 35
Naučeni NEXT: 03 13 21 23 32 35 37
Promena Born verovatnoća zbog istorije: 7.245344677e-02
Promena uslovnih margina zbog istorije: 6.562299158e-02
Zbir uslovnih margina: 7.000000000000

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.03 | prelaza=1039
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.8 s
Iteracija 50 | proteklo 1.5 s
Iteracija 75 | proteklo 2.3 s
Iteracija 100 | proteklo 3.1 s
Iteracija 125 | proteklo 3.8 s
Iteracija 150 | proteklo 4.7 s
Prihvaćen lokalni minimum: 16.576239858
Max|grad|: 8.581e-07
Trajanje: 5.4 s
Validacioni NLL: 16.801207645

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.03 | prelaza=1212
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.8 s
Iteracija 50 | proteklo 1.6 s
Iteracija 75 | proteklo 2.5 s
Iteracija 100 | proteklo 3.3 s
Iteracija 125 | proteklo 4.2 s
Iteracija 150 | proteklo 5.1 s
Iteracija 175 | proteklo 5.9 s
Prihvaćen lokalni minimum: 16.592856692
Max|grad|: 8.443e-07
Trajanje: 6.6 s
Validacioni NLL: 16.586909433
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.694058539

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.3 | prelaza=1039
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.8 s
Iteracija 50 | proteklo 1.6 s
Iteracija 75 | proteklo 2.5 s
Iteracija 100 | proteklo 3.3 s
Iteracija 125 | proteklo 4.2 s
Iteracija 150 | proteklo 5.0 s
Iteracija 175 | proteklo 5.8 s
Prihvaćen lokalni minimum: 16.936596628
Max|grad|: 6.734e-07
Trajanje: 5.9 s
Validacioni NLL: 16.937741397

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.3 | prelaza=1212
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.9 s
Iteracija 50 | proteklo 1.7 s
Iteracija 75 | proteklo 2.5 s
Iteracija 100 | proteklo 3.4 s
Iteracija 125 | proteklo 4.3 s
Iteracija 150 | proteklo 5.2 s
Iteracija 175 | proteklo 6.0 s
Prihvaćen lokalni minimum: 16.974842869
Max|grad|: 8.828e-07
Trajanje: 6.7 s
Validacioni NLL: 16.635921951
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 16.786831674

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.03 | prelaza=1039
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.9 s
Iteracija 50 | proteklo 1.8 s
Iteracija 75 | proteklo 2.6 s
Iteracija 100 | proteklo 3.4 s
Iteracija 125 | proteklo 4.2 s
Iteracija 150 | proteklo 5.0 s
Iteracija 175 | proteklo 5.8 s
Iteracija 200 | proteklo 6.7 s
Iteracija 225 | proteklo 7.5 s
Prihvaćen lokalni minimum: 16.489958159
Max|grad|: 7.821e-07
Trajanje: 8.2 s
Validacioni NLL: 16.671464918

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.03 | prelaza=1212
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.0 s
Iteracija 50 | proteklo 1.9 s
Iteracija 75 | proteklo 2.8 s
Iteracija 100 | proteklo 3.6 s
Iteracija 125 | proteklo 4.6 s
Iteracija 150 | proteklo 5.6 s
Iteracija 175 | proteklo 6.6 s
Prihvaćen lokalni minimum: 16.506135241
Max|grad|: 8.389e-07
Trajanje: 7.4 s
Validacioni NLL: 16.612216657
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.641840788

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.3 | prelaza=1039
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 0.9 s
Iteracija 50 | proteklo 1.7 s
Iteracija 75 | proteklo 2.6 s
Iteracija 100 | proteklo 3.5 s
Iteracija 125 | proteklo 4.3 s
Iteracija 150 | proteklo 5.1 s
Iteracija 175 | proteklo 5.9 s
Prihvaćen lokalni minimum: 16.760235594
Max|grad|: 8.403e-07
Trajanje: 6.3 s
Validacioni NLL: 16.750491593

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.3 | prelaza=1212
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.0 s
Iteracija 50 | proteklo 2.0 s
Iteracija 75 | proteklo 2.9 s
Iteracija 100 | proteklo 3.8 s
Iteracija 125 | proteklo 4.7 s
Iteracija 150 | proteklo 5.5 s
Prihvaćen lokalni minimum: 16.779192329
Max|grad|: 7.816e-07
Trajanje: 5.8 s
Validacioni NLL: 16.667740158
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.709115875

IZABRANO: lag=3 | λ=0.03
Validacioni NLL: 16.641840788

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=3 | λ=0.03 | prelaza=1385
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.1 s
Iteracija 50 | proteklo 2.0 s
Iteracija 75 | proteklo 3.0 s
Iteracija 100 | proteklo 3.9 s
Iteracija 125 | proteklo 4.9 s
Iteracija 150 | proteklo 5.8 s
Iteracija 175 | proteklo 6.7 s
Iteracija 200 | proteklo 7.6 s
Iteracija 225 | proteklo 8.6 s
Iteracija 250 | proteklo 9.6 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 2.924e-06
Prihvaćen lokalni minimum: 16.514708468
Max|grad|: 9.103e-07
Trajanje: 9.9 s
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 16.621520268

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=3 | λ=0.03 | prelaza=1732
Kvantnih ugaonih parametara: 208
Iteracija 25 | proteklo 1.1 s
Iteracija 50 | proteklo 2.3 s
Iteracija 75 | proteklo 3.3 s
Iteracija 100 | proteklo 4.4 s
Iteracija 125 | proteklo 5.5 s
Iteracija 150 | proteklo 6.6 s
Iteracija 175 | proteklo 7.7 s
Iteracija 200 | proteklo 8.8 s
Iteracija 225 | proteklo 9.8 s
Iteracija 250 | proteklo 10.9 s
Nastavak 1
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Max|grad|: 9.751e-06
Iteracija 275 | proteklo 12.1 s
Prihvaćen lokalni minimum: 16.526982978
Max|grad|: 8.130e-07
Trajanje: 13.0 s

ZAVRŠNO UČENJE: svih 1732 prelaza
Početni EARAC maksimum: 03 10 11 20 26 32 34
Naučeni NEXT: 11 18 26 27 32 33 34
Promena Born verovatnoća zbog istorije: 1.297438948e-01
Promena uslovnih margina zbog istorije: 1.128766831e-01
Zbir uslovnih margina: 7.000000000000



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 08 13 16 21 26 34 39
Validacioni NLL: 16.584856356
Odvojeni test NLL: 16.571771674
Modelna verovatnoća NEXT: 1.77182723894e-07
Zbir uslovnih margina: 7.000000000000
Osetljivost uslovnih margina: 6.247465902e-02

loto7_4710_k86_loto_2977.csv
NEXT: 03 13 21 23 32 35 37
Validacioni NLL: 16.602117820
Odvojeni test NLL: 16.594303055
Modelna verovatnoća NEXT: 1.87354082132e-07
Zbir uslovnih margina: 7.000000000000
Osetljivost uslovnih margina: 6.562299158e-02

loto7_4710_k86_loto_plus_1733.csv
NEXT: 11 18 26 27 32 33 34
Validacioni NLL: 16.641840788
Odvojeni test NLL: 16.621520268
Modelna verovatnoća NEXT: 3.54699092807e-07
Zbir uslovnih margina: 7.000000000000
Osetljivost uslovnih margina: 1.128766831e-01

Obrada je završena.
NEXT je tačan maksimum raspodele za naučene parametre.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
Izlaz dolazi iz Bornovih verovatnoća Qiskit Alice-Bob protokola, a učenje će podešavati uglove Aliceinih kvantnih merenja. 
Za ovaj oblik raspodele maksimum može da se odredi tačno bez pretrage 15 miliona sedmorki; tu jednakost će proveravati mala potpuna enumeracija.
Nema klasičnih prediktivnih grana ni regresora iza kvantnih karakteristika. 
Qiskit priprema singlet, primenjuje Aliceino naučeno merenje i računa Bornove verovatnoće Bobovog odgovora. 
Iz njih se izvodi normalizovana raspodela sedmorki.
Trobitna Alice-Bob ideja. Svako očitavanje koristi zasebnu pripremu stanja. 
Klasični deo obavlja optimizaciju, uslovljavanje na sedam brojeva i numeričke provere.
"""
