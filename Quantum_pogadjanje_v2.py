#!/usr/bin/env python3

# Quantum_pogadjanje


"""
QUANTUM POGADJANJE V2 — Qiskit EARAC — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Preserved quantum construction:
    history controls Alice's Rx/Ry/Rz angles;
    Bob learns an orthogonal measurement frame;
    Alice and Bob share a two-qubit singlet;
    Alice sends one classical bit;
    Born probabilities define the decoded outputs.

Training uses vectorized complex-amplitude propagation.
Self-checks compare it with explicit Qiskit Statevector circuits.
Final NEXT probabilities use explicit Qiskit circuits.

Optimizer correction:
    retain the best numerically valid evaluated point;
    iteration limits do not imply invalid probabilities;
    report convergence status separately;
    no automatic threefold iteration-budget extension.

Persistence:
    save best parameters during fitting;
    save completed fitting jobs;
    reuse matching checkpoints on the next invocation;
    input contents and configuration determine checkpoint identity.

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
from itertools import combinations
from pathlib import Path

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBER_COUNT = 39
DRAW_SIZE = 7
GROUP_COUNT = 13

CANDIDATES = (
    (1, 0.001),
    (1, 0.010),
    (3, 0.001),
    (3, 0.010),
)

INITIAL_ORDERS = (17, 11)

OPTIMIZER_STEPS = 600
GRADIENT_LIMIT = 5e-4
BATCH_SIZE = 512

CHECKPOINT_VERSION = "EARAC_V2_AMPLITUDES_BEST_VALID_20261029"
CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "Quantum_pogadjanje_v2_checkpoints"
)

IDENTITY = np.eye(2, dtype=np.complex128)

GENERATORS = tuple(
    np.asarray(Pauli(label).to_matrix(), dtype=np.complex128)
    for label in ("X", "Y", "Z")
)

HADAMARD = np.array(
    [[1.0, 1.0], [1.0, -1.0]],
    dtype=np.complex128,
) / math.sqrt(2.0)

SDG = np.diag(
    np.array([1.0, -1j], dtype=np.complex128)
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rotate_vectors(vectors, angles, with_jacobian=False):
    vectors, angles = np.broadcast_arrays(
        np.asarray(vectors, dtype=float),
        np.asarray(angles, dtype=float),
    )

    require(
        vectors.shape[-1] == 3
        and np.all(np.isfinite(vectors))
        and np.all(np.isfinite(angles)),
        "Neispravan ulaz kvantnih rotacija.",
    )

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    sx = np.sin(angles[..., 0])
    cx = np.cos(angles[..., 0])
    sy = np.sin(angles[..., 1])
    cy = np.cos(angles[..., 1])
    sz = np.sin(angles[..., 2])
    cz = np.cos(angles[..., 2])

    u0 = x
    u1 = cx * y - sx * z
    u2 = sx * y + cx * z

    v0 = cy * u0 + sy * u2
    v1 = u1
    v2 = -sy * u0 + cy * u2

    rotated = np.stack(
        (
            cz * v0 - sz * v1,
            sz * v0 + cz * v1,
            v2,
        ),
        axis=-1,
    )

    if not with_jacobian:
        return rotated, None

    dx_u1 = -sx * y - cx * z
    dx_u2 = cx * y - sx * z

    dx_v0 = sy * dx_u2
    dx_v1 = dx_u1
    dx_v2 = cy * dx_u2

    derivative_x = np.stack(
        (
            cz * dx_v0 - sz * dx_v1,
            sz * dx_v0 + cz * dx_v1,
            dx_v2,
        ),
        axis=-1,
    )

    dy_v0 = -sy * u0 + cy * u2
    dy_v2 = -cy * u0 - sy * u2

    derivative_y = np.stack(
        (
            cz * dy_v0,
            sz * dy_v0,
            dy_v2,
        ),
        axis=-1,
    )

    derivative_z = np.stack(
        (
            -sz * v0 - cz * v1,
            cz * v0 - sz * v1,
            np.zeros_like(v0),
        ),
        axis=-1,
    )

    return rotated, np.stack(
        (derivative_x, derivative_y, derivative_z),
        axis=-2,
    )


def rotation_gate(axis, angles):
    angles = np.asarray(angles, dtype=float)

    return (
        np.cos(angles / 2.0)[..., None, None] * IDENTITY
        - 1j
        * np.sin(angles / 2.0)[..., None, None]
        * GENERATORS[axis]
    )


def apply_gate(states, gate, qubit):
    tensor = states.reshape(states.shape[:-1] + (2, 2))

    if qubit == 0:
        result = np.einsum(
            "...ab,...ib->...ia",
            gate,
            tensor,
            optimize=True,
        )
    elif qubit == 1:
        result = np.einsum(
            "...ab,...bj->...aj",
            gate,
            tensor,
            optimize=True,
        )
    else:
        raise RuntimeError("Neispravan indeks kubita.")

    return result.reshape(result.shape[:-2] + (4,))


def unpack(parameters, feature_count, groups):
    size = feature_count * groups * 3

    alice = parameters[:size].reshape(
        feature_count, groups, 3
    )

    bob = parameters[size:].reshape(groups, 3)

    return alice, bob


def pack(alice, bob):
    return np.concatenate((alice.ravel(), bob.ravel()))


class QuantumProtocol:
    def __init__(self):
        circuit = QuantumCircuit(2, name="EARACSinglet")

        circuit.x(0)
        circuit.h(1)
        circuit.cx(1, 0)
        circuit.z(1)

        self.singlet_circuit = circuit
        self.initial_amplitudes = np.asarray(
            Statevector.from_instruction(circuit).data,
            dtype=np.complex128,
        )

        require(
            abs(float(np.vdot(
                self.initial_amplitudes,
                self.initial_amplitudes,
            ).real) - 1.0) < 1e-12,
            "Singlet nije normalizovan.",
        )

    @staticmethod
    def baseline_data(latest_bits):
        n, m = latest_bits.shape

        require(
            m % 3 == 0
            and np.all(
                (latest_bits == 0) | (latest_bits == 1)
            ),
            "Neispravne istorijske trobitne grupe.",
        )

        triples = latest_bits.reshape(n, m // 3, 3)
        baseline = (1.0 - 2.0 * triples) / math.sqrt(3.0)

        first_bit = triples[:, :, 0].astype(np.int64)
        first_sign = 1.0 - 2.0 * first_bit
        alice_axis = first_sign[:, :, None] * baseline

        polar = np.arctan2(
            np.sqrt(
                alice_axis[:, :, 0] ** 2
                + alice_axis[:, :, 1] ** 2
            ),
            alice_axis[:, :, 2],
        )

        azimuth = np.arctan2(
            alice_axis[:, :, 1],
            alice_axis[:, :, 0],
        )

        return baseline, first_bit, polar, azimuth

    def vectorized_states(self, alice_angles, bob_angles, latest_bits):
        n, m = latest_bits.shape
        groups = m // 3

        _, _, polar, azimuth = self.baseline_data(latest_bits)

        states = np.broadcast_to(
            self.initial_amplitudes,
            (n, groups, 4),
        ).copy()

        for axis in (2, 1, 0):
            states = apply_gate(
                states,
                rotation_gate(axis, -alice_angles[:, :, axis]),
                0,
            )

        states = apply_gate(states, rotation_gate(2, -azimuth), 0)
        states = apply_gate(states, rotation_gate(1, -polar), 0)

        for axis in (2, 1, 0):
            states = apply_gate(
                states,
                rotation_gate(axis, -bob_angles[None, :, axis]),
                1,
            )

        outputs = []

        for query in range(3):
            queried = states

            if query == 0:
                queried = apply_gate(queried, HADAMARD, 1)
            elif query == 1:
                queried = apply_gate(queried, SDG, 1)
                queried = apply_gate(queried, HADAMARD, 1)

            outputs.append(queried)

        return np.stack(outputs, axis=2)

    def circuit(self, alice_angles, bob_angles, latest_triple, query):
        triple = np.asarray(latest_triple, dtype=float)

        _, _, polar, azimuth = self.baseline_data(
            triple.reshape(1, 3)
        )

        circuit = self.singlet_circuit.copy()

        circuit.rz(-float(alice_angles[2]), 0)
        circuit.ry(-float(alice_angles[1]), 0)
        circuit.rx(-float(alice_angles[0]), 0)

        circuit.rz(-float(azimuth[0, 0]), 0)
        circuit.ry(-float(polar[0, 0]), 0)

        circuit.rz(-float(bob_angles[2]), 1)
        circuit.ry(-float(bob_angles[1]), 1)
        circuit.rx(-float(bob_angles[0]), 1)

        if query == 0:
            circuit.h(1)
        elif query == 1:
            circuit.sdg(1)
            circuit.h(1)
        elif query != 2:
            raise RuntimeError("Neispravan Bobov upit.")

        return circuit

    @staticmethod
    def decode(states, latest_bits):
        n, m = latest_bits.shape
        groups = m // 3

        first_bit = latest_bits.reshape(
            n, groups, 3
        )[:, :, 0].astype(np.int64)

        joint = np.abs(states) ** 2

        require(
            np.all(np.isfinite(joint))
            and np.max(np.abs(
                joint.sum(axis=-1) - 1.0
            )) < 2e-9,
            "Born verovatnoće nisu normalizovane.",
        )

        q = np.zeros((n, groups, 3))
        r = np.zeros_like(q)

        for index in range(4):
            alice_outcome = index & 1
            bob_outcome = (index >> 1) & 1

            message = first_bit ^ alice_outcome
            guess = message ^ (1 - bob_outcome)

            q += joint[:, :, :, index] * guess[:, :, None]
            r += joint[:, :, :, index] * (1 - guess[:, :, None])

        total = q + r

        require(
            np.max(np.abs(total - 1.0)) < 2e-9,
            "Dekodirane verovatnoće nisu normalizovane.",
        )

        q /= total
        r /= total

        require(
            np.all(np.isfinite(q))
            and np.all(np.isfinite(r))
            and np.all(q > 0)
            and np.all(r > 0),
            "Nulta ili nevažeća EARAC verovatnoća.",
        )

        return q.reshape(n, m), r.reshape(n, m)

    def probabilities(
        self,
        alice_weights,
        bob_angles,
        x,
        latest_bits,
        with_jacobian=False,
        explicit=False,
    ):
        n, m = latest_bits.shape
        groups = m // 3

        require(
            alice_weights.shape == (x.shape[1], groups, 3)
            and bob_angles.shape == (groups, 3),
            "Pogrešan oblik kvantnih parametara.",
        )

        alice_angles = np.einsum(
            "nf,fga->nga",
            x,
            alice_weights,
            optimize=True,
        )

        states = self.vectorized_states(
            alice_angles, bob_angles, latest_bits
        )

        circuit_error = None

        if explicit:
            actual = np.empty_like(states)
            triples = latest_bits.reshape(n, groups, 3)

            for sample in range(n):
                for group in range(groups):
                    for query in range(3):
                        circuit = self.circuit(
                            alice_angles[sample, group],
                            bob_angles[group],
                            triples[sample, group],
                            query,
                        )

                        actual[sample, group, query] = (
                            Statevector.from_instruction(circuit).data
                        )

            circuit_error = float(
                np.max(np.abs(actual - states))
            )

            require(
                circuit_error < 2e-9,
                "Amplitude nisu saglasne sa Qiskitom:"
                f" {circuit_error:.3e}",
            )

            states = actual

        q, r = self.decode(states, latest_bits)
        baseline, _, _, _ = self.baseline_data(latest_bits)

        corrected, alice_jacobian = rotate_vectors(
            baseline,
            alice_angles,
            with_jacobian,
        )

        bob_axes, bob_jacobian = rotate_vectors(
            np.broadcast_to(np.eye(3), (groups, 3, 3)),
            np.broadcast_to(
                bob_angles[:, None, :],
                (groups, 3, 3),
            ),
            with_jacobian,
        )

        reference = (
            1.0
            - np.einsum(
                "ngc,gjc->ngj",
                corrected,
                bob_axes,
                optimize=True,
            )
        ) / 2.0

        require(
            np.max(np.abs(
                q.reshape(n, groups, 3) - reference
            )) < 2e-9,
            "Amplitude nisu saglasne sa singlet konstrukcijom.",
        )

        bias = (r - q).reshape(n, groups, 3)

        require(
            np.max(np.abs(
                np.sum(bias * bias, axis=2) - 1.0
            )) < 2e-8,
            "Narušena je sferna kvantna granica.",
        )

        return (
            q, r, corrected, alice_jacobian,
            bob_axes, bob_jacobian, circuit_error,
        )


PROTOCOL = QuantumProtocol()


def fixed_size_partition(fields, k, with_marginals=False):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 2
        and np.all(np.isfinite(fields))
        and 1 <= k <= fields.shape[1],
        "Neispravan ulaz normalizacije.",
    )

    n, m = fields.shape

    forward = np.full((n, m + 1, k + 1), -np.inf)
    forward[:, 0, 0] = 0.0

    for number in range(m):
        forward[:, number + 1, 0] = 0.0
        forward[:, number + 1, 1:] = np.logaddexp(
            forward[:, number, 1:],
            fields[:, number, None] + forward[:, number, :-1],
        )

    log_z = forward[:, m, k]

    require(
        np.all(np.isfinite(log_z)),
        "Normalizacija nije konačna.",
    )

    if not with_marginals:
        return log_z, None

    backward = np.full_like(forward, -np.inf)
    backward[:, m, 0] = 0.0

    for number in range(m - 1, -1, -1):
        backward[:, number, 0] = 0.0
        backward[:, number, 1:] = np.logaddexp(
            backward[:, number + 1, 1:],
            fields[:, number, None] + backward[:, number + 1, :-1],
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
        "Uslovne margine nisu ispravne.",
    )

    require(
        np.max(np.abs(
            marginals.sum(axis=1) - k
        )) < 1e-7,
        "Zbir uslovnih margina nije veličina kombinacije.",
    )

    return log_z, marginals


def objective(
    parameters,
    x,
    latest_bits,
    targets,
    ridge,
    k,
    gradient=True,
):
    groups = latest_bits.shape[1] // 3

    alice_weights, bob_angles = unpack(
        parameters, x.shape[1], groups
    )

    n = len(targets)
    total_nll = 0.0

    gradient_alice = np.zeros_like(alice_weights)
    gradient_bob = np.zeros_like(bob_angles)

    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)

        batch_x = x[start:stop]
        batch_bits = latest_bits[start:stop]
        batch_targets = targets[start:stop]

        (
            q, r, corrected, alice_jacobian,
            bob_axes, bob_jacobian, _,
        ) = PROTOCOL.probabilities(
            alice_weights,
            bob_angles,
            batch_x,
            batch_bits,
            with_jacobian=gradient,
        )

        fields = np.log(q) - np.log(r)

        log_z, marginals = fixed_size_partition(
            fields, k, with_marginals=gradient
        )

        observed = np.zeros_like(q)
        observed[
            np.arange(len(batch_targets))[:, None],
            batch_targets,
        ] = 1.0

        logp = np.sum(fields * observed, axis=1) - log_z

        require(
            np.all(np.isfinite(logp))
            and np.max(logp) <= 1e-7,
            "Neispravna verovatnoća posmatranog kola.",
        )

        total_nll -= float(logp.sum())

        if gradient:
            gradient_q = (
                (marginals - observed) / (q * r)
            ).reshape(len(batch_targets), groups, 3)

            gradient_bloch = -0.5 * np.einsum(
                "ngj,gjc->ngc",
                gradient_q, bob_axes,
                optimize=True,
            )

            gradient_bob_axes = -0.5 * np.einsum(
                "ngj,ngc->gjc",
                gradient_q, corrected,
                optimize=True,
            )

            gradient_angles = np.einsum(
                "ngc,ngac->nga",
                gradient_bloch, alice_jacobian,
                optimize=True,
            )

            gradient_alice += np.einsum(
                "nf,nga->fga",
                batch_x, gradient_angles,
                optimize=True,
            )

            gradient_bob += np.einsum(
                "gjc,gjac->ga",
                gradient_bob_axes, bob_jacobian,
                optimize=True,
            )

    penalty_mask = np.ones(x.shape[1])
    penalty_mask[0] = 0.1

    penalty = ridge * (
        np.sum(
            alice_weights ** 2
            * penalty_mask[:, None, None]
        )
        + np.sum(bob_angles ** 2)
    )

    loss = total_nll / n + float(penalty)

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    gradient_alice = (
        gradient_alice / n
        + 2.0 * ridge * alice_weights
        * penalty_mask[:, None, None]
    )

    gradient_bob = (
        gradient_bob / n + 2.0 * ridge * bob_angles
    )

    result = pack(gradient_alice, gradient_bob)

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


def initial_parameters(feature_count, groups, order):
    weights = np.zeros((feature_count, groups, 3))

    phase = (
        2.0 * math.pi
        * (order * np.arange(groups) + 1)
        / (groups + 1)
    )

    for axis in range(3):
        weights[0, :, axis] = (
            0.025 * np.sin(phase + 0.7 * axis)
        )

        if feature_count > 1:
            weights[1, :, axis] = (
                0.015 * np.cos(phase + 0.9 * axis)
            )

    bob = np.column_stack(
        [
            0.02 * np.cos(phase + 0.5 * axis)
            for axis in range(3)
        ]
    )

    return pack(weights, bob)


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


def make_context(occupancy, targets, lookback):
    targets = np.asarray(targets, dtype=np.int64)

    require(
        targets.ndim == 1
        and np.all(targets >= 1)
        and np.all(targets <= len(occupancy)),
        "Neispravni indeksi istorijskih konteksta.",
    )

    m = occupancy.shape[1]
    width = m + 1

    raw = np.zeros((len(targets), lookback * width))

    for lag in range(1, lookback + 1):
        previous = targets - lag
        valid = np.flatnonzero(previous >= 0)
        offset = (lag - 1) * width

        if len(valid):
            raw[valid, offset:offset + m] = occupancy[previous[valid]]
            raw[valid, offset + m] = 1.0

    return raw, occupancy[targets - 1].copy()


def fit_scaling(raw):
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0

    return mean, scale


def transform_context(raw, mean, scale):
    return np.column_stack(
        (np.ones(len(raw)), (raw - mean) / scale)
    )


def checkpoint_path(raw, latest, targets, lookback, ridge, order):
    digest = hashlib.sha256()

    configuration = (
        CHECKPOINT_VERSION,
        int(lookback),
        float(ridge),
        int(order),
        OPTIMIZER_STEPS,
        qiskit.__version__,
        np.__version__,
    )

    digest.update(repr(configuration).encode("utf-8"))

    for array in (raw, latest, targets):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())

    return CHECKPOINT_DIR / (digest.hexdigest() + ".npz")


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(path.name + ".tmp")

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


def load_checkpoint(path, expected_shape):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            parameters = saved["parameters"].astype(float)
            iterations = int(saved["iterations"].item())
            done = bool(saved["done"].item())
            message = str(saved["message"].item())

        require(
            parameters.shape == expected_shape
            and np.all(np.isfinite(parameters))
            and iterations >= 0,
            "Neispravna kontrolna tačka.",
        )

        return parameters, iterations, done, message

    except (OSError, ValueError, KeyError, RuntimeError) as error:
        say(
            f"Kontrolna tačka nije učitana: {path.name}",
            str(error),
        )

        return None


@dataclass
class Model:
    alice_weights: np.ndarray
    bob_angles: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    lookback: int
    ridge: float
    train_loss: float
    gradient_norm: float
    converged: bool
    optimizer_message: str

    def transform(self, raw):
        return transform_context(raw, self.mean, self.scale)

    def probabilities(self, raw, latest, explicit=False):
        result = PROTOCOL.probabilities(
            self.alice_weights,
            self.bob_angles,
            self.transform(raw),
            latest,
            explicit=explicit,
        )

        return result[0], result[1], result[-1]


def fit_model(raw, latest, targets, lookback, ridge, label):
    mean, scale = fit_scaling(raw)
    x = transform_context(raw, mean, scale)
    groups = latest.shape[1] // 3

    accepted = []

    for order in INITIAL_ORDERS:
        initial = initial_parameters(x.shape[1], groups, order)

        path = checkpoint_path(
            raw, latest, targets, lookback, ridge, order
        )

        saved = load_checkpoint(path, initial.shape)

        parameters = initial.copy()
        used_iterations = 0
        already_done = False
        message = "Novi proračun."

        if saved is not None:
            parameters, used_iterations, already_done, message = saved

        started = time.perf_counter()

        say(
            "",
            label,
            f"Lag={lookback} | λ={ridge:g}"
            f" | početak={order} | prelaza={len(targets)}",
        )

        try:
            # Establish a numerically valid retained point before optimization.
            best_loss, best_gradient = objective(
                parameters, x, latest, targets, ridge, DRAW_SIZE
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            if saved is None:
                say(f"Početak {order} nije numerički validan:", str(error))
                continue

            say(
                "Sačuvana tačka nije prošla ponovnu proveru.",
                "Koristim deterministički početak.",
            )

            parameters = initial.copy()
            used_iterations = 0
            already_done = False

            best_loss, best_gradient = objective(
                parameters, x, latest, targets, ridge, DRAW_SIZE
            )

        best_parameters = parameters.copy()
        best_gradient = best_gradient.copy()
        evaluations = 0
        added_iterations = 0
        optimizer_success = False

        if already_done:
            say("Učitan završen proračun iz kontrolne tačke.")

        else:
            remaining = max(0, OPTIMIZER_STEPS - used_iterations)

            # Write before a long run, so a permissions problem appears early.
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
                nonlocal evaluations

                loss, gradient = objective(
                    current, x, latest, targets, ridge, DRAW_SIZE
                )

                evaluations += 1

                if loss < best_loss:
                    best_loss = float(loss)
                    best_parameters = current.copy()
                    best_gradient = gradient.copy()

                return loss, gradient

            def callback(current):
                nonlocal added_iterations
                added_iterations += 1

                if added_iterations % 25 == 0:
                    total_iterations = used_iterations + added_iterations

                    save_checkpoint(
                        path,
                        best_parameters,
                        total_iterations,
                        False,
                        "Sačuvana najbolja validna tačka.",
                    )

                    say(
                        f"Iteracija {total_iterations}"
                        f" | proteklo {time.perf_counter() - started:.1f} s",
                        f"Najbolji cilj: {best_loss:.9f}",
                        "Max|grad| te tačke:"
                        f" {np.max(np.abs(best_gradient)):.3e}",
                    )

            if remaining > 0:
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

                    # Evaluate the returned point as well.
                    function(result.x)

                except (
                    RuntimeError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as error:
                    message = (
                        "Pretraga je prekinuta; zadržana je prethodna "
                        "validna tačka: " + str(error)
                    )

            else:
                message = (
                    "Dostignut sačuvani budžet iteracija; "
                    "koristi se najbolja validna tačka."
                )

            used_iterations += added_iterations

        # Revalidate the retained point on the entire fitting set.
        checked_loss, checked_gradient = objective(
            best_parameters,
            x,
            latest,
            targets,
            ridge,
            DRAW_SIZE,
        )

        require(
            np.isfinite(checked_loss)
            and np.all(np.isfinite(checked_gradient)),
            "Sačuvana tačka nije numerički ispravna.",
        )

        maximum_gradient = float(
            np.max(np.abs(checked_gradient))
        )

        alice, bob = unpack(
            best_parameters, x.shape[1], groups
        )

        # Small explicit Qiskit check before accepting this fitted model.
        PROTOCOL.probabilities(
            alice,
            bob,
            x[:min(2, len(x))],
            latest[:min(2, len(latest))],
            explicit=True,
        )

        converged = (
            maximum_gradient <= GRADIENT_LIMIT
        )

        if not already_done:
            status_prefix = (
                "OPTIMIZER_SUCCESS"
                if optimizer_success
                else "OPTIMIZER_NOT_CONFIRMED"
            )

            message = status_prefix + ": " + message

            save_checkpoint(
                path,
                best_parameters,
                used_iterations,
                True,
                message,
            )

        model = Model(
            alice_weights=alice.copy(),
            bob_angles=bob.copy(),
            mean=mean.copy(),
            scale=scale.copy(),
            lookback=lookback,
            ridge=ridge,
            train_loss=float(checked_loss),
            gradient_norm=maximum_gradient,
            converged=converged,
            optimizer_message=message,
        )

        accepted.append(model)

        say(
            f"Zadržana validna tačka: {checked_loss:.9f}",
            f"Max|grad|: {maximum_gradient:.3e}",
            "Gradijentni uslov: "
            + ("dostignut" if converged else "NIJE dostignut"),
            message,
            f"Trajanje ovog poziva: {time.perf_counter() - started:.1f} s",
        )

        if not converged:
            say(
                "DIJAGNOSTIKA: konvergencija nije potvrđena.",
                "Tačka nije proglašena lokalnim minimumom.",
                "Validacioni postupak nastavlja sa njenom ispravnom raspodelom.",
            )

    require(
        bool(accepted),
        f"{label}: nema nijedne numerički ispravne tačke.",
    )

    # Select deterministic initialization by training objective only.
    return min(accepted, key=lambda model: model.train_loss)


def evaluate_nll(model, raw, latest, targets):
    total = 0.0

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))

        q, r, _ = model.probabilities(
            raw[start:stop], latest[start:stop]
        )

        fields = np.log(q) - np.log(r)

        log_z, _ = fixed_size_partition(fields, DRAW_SIZE)

        scores = fields[
            np.arange(stop - start)[:, None],
            targets[start:stop],
        ].sum(axis=1)

        logp = scores - log_z

        require(
            np.all(np.isfinite(logp))
            and np.max(logp) <= 1e-7,
            "Neispravna evaluaciona verovatnoća.",
        )

        total -= float(logp.sum())

    return total / len(targets)


def exact_next(fields, k):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 1
        and np.all(np.isfinite(fields))
        and 1 <= k <= len(fields),
        "Neispravna polja za maksimum.",
    )

    order = sorted(
        range(len(fields)),
        key=lambda number: (-float(fields[number]), number),
    )

    return tuple(sorted(order[:k]))


def self_checks():
    say("", "Pokrećem ugrađene Qiskit i matematičke provere.")

    started = time.perf_counter()

    x = np.array(
        [[1.0, 0.2, -0.3], [1.0, -0.4, 0.5], [1.0, 0.6, 0.1]]
    )

    bits = np.array(
        [
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=float,
    )

    targets = np.array(
        [[0, 4], [1, 3], [2, 5]],
        dtype=np.int64,
    )

    k = 2
    ridge = 0.01

    parameters = initial_parameters(x.shape[1], 2, 17)
    alice, bob = unpack(parameters, x.shape[1], 2)

    explicit = PROTOCOL.probabilities(
        alice, bob, x, bits,
        with_jacobian=True,
        explicit=True,
    )

    all_bits = np.array(
        [
            [(code >> 2) & 1, (code >> 1) & 1, code & 1]
            for code in range(8)
        ],
        dtype=float,
    )

    zero_result = PROTOCOL.probabilities(
        np.zeros((1, 1, 3)),
        np.zeros((1, 3)),
        np.ones((8, 1)),
        all_bits,
        explicit=True,
    )

    q0, r0 = zero_result[:2]

    require(
        np.max(np.abs(
            np.where(all_bits == 1, q0, r0)
            - (1.0 + 1.0 / math.sqrt(3.0)) / 2.0
        )) < 1e-11,
        "Originalni EARAC protokol nije prošao.",
    )

    _, analytic = objective(
        parameters, x, bits, targets, ridge, k
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
                positive, x, bits, targets, ridge, k,
                gradient=False,
            )
            - objective(
                negative, x, bits, targets, ridge, k,
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

    q, r = explicit[:2]
    fields = np.log(q) - np.log(r)

    log_z, marginals = fixed_size_partition(
        fields, k, with_marginals=True
    )

    subsets = list(combinations(range(6), k))
    indices = np.asarray(subsets, dtype=np.int64)

    for sample in range(len(x)):
        scores = fields[sample][indices].sum(axis=1)
        direct_log_z = float(logsumexp(scores))
        probabilities = np.exp(scores - direct_log_z)

        require(
            abs(direct_log_z - log_z[sample]) < 1e-11
            and abs(float(probabilities.sum()) - 1.0) < 1e-12,
            "Provera normalizacije nije prošla.",
        )

        direct_marginals = np.zeros(6)

        for selected, probability in zip(subsets, probabilities):
            direct_marginals[np.asarray(selected)] += probability

        require(
            np.max(np.abs(
                direct_marginals - marginals[sample]
            )) < 1e-11,
            "Provera margina nije prošla.",
        )

        require(
            exact_next(fields[sample], k)
            == subsets[int(np.argmax(scores))],
            "Provera tačnog maksimuma nije prošla.",
        )

    for boundary_k in (1, 6):
        z, marginal = fixed_size_partition(
            np.zeros((1, 6)),
            boundary_k,
            with_marginals=True,
        )

        require(
            abs(z[0] - math.log(math.comb(6, boundary_k))) < 1e-11
            and np.max(np.abs(
                marginal[0] - boundary_k / 6
            )) < 1e-11,
            "Granična provera nije prošla.",
        )

    require(
        exact_next(np.zeros(6), 2) == (0, 1),
        "Obrada jednakih maksimuma nije prošla.",
    )

    toy_raw, toy_latest = make_context(bits, [1], 3)

    require(
        np.array_equal(toy_raw[0, :6], bits[0])
        and toy_raw[0, 6] == 1
        and np.all(toy_raw[0, 7:] == 0)
        and np.array_equal(toy_latest[0], bits[0]),
        "Prvi istorijski prelaz nije ispravan.",
    )

    say(
        "Ugrađene provere su prošle.",
        f"Greška amplituda prema Qiskitu: {explicit[-1]:.3e}",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
        f"Trajanje provera: {time.perf_counter() - started:.1f} s",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def process_file(path, draws, occupancy, fingerprint):
    targets = draws[1:].copy()
    n = len(targets)

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno istorije za hronološku podelu.",
    )

    contexts = {
        lag: make_context(
            occupancy, np.arange(1, len(draws)), lag
        )
        for lag in sorted({item[0] for item in CANDIDATES})
    }

    folds = ((split60, split70), (split70, split80))
    scores = []

    for lookback, ridge in CANDIDATES:
        raw, latest = contexts[lookback]

        total_nll = 0.0
        total_count = 0
        incomplete_folds = 0

        for fold, (train_end, validation_end) in enumerate(folds, 1):
            model = fit_model(
                raw[:train_end],
                latest[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                f"{path.name} / validacija {fold}",
            )

            incomplete_folds += int(not model.converged)

            score = evaluate_nll(
                model,
                raw[train_end:validation_end],
                latest[train_end:validation_end],
                targets[train_end:validation_end],
            )

            count = validation_end - train_end
            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count

        scores.append(
            (score, lookback, ridge, incomplete_folds)
        )

        say(
            f"KANDIDAT: lag={lookback} | λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
            f"Podela bez potvrđenog gradijentnog uslova: {incomplete_folds}",
        )

    validation_nll, lookback, ridge, incomplete_folds = min(
        scores, key=lambda item: item[0]
    )

    raw, latest = contexts[lookback]

    say(
        "",
        f"IZABRANO: lag={lookback} | λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        raw[:split80],
        latest[:split80],
        targets[:split80],
        lookback,
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        raw[split80:],
        latest[split80:],
        targets[split80:],
    )

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"NLL cele sedmorke: {test_nll:.9f}",
    )

    final_model = fit_model(
        raw,
        latest,
        targets,
        lookback,
        ridge,
        f"{path.name} / završno učenje",
    )

    raw_next, latest_next = make_context(
        occupancy, [len(draws)], lookback
    )

    q, r, circuit_error = final_model.probabilities(
        raw_next, latest_next, explicit=True
    )

    fields = np.log(q) - np.log(r)

    log_z, inclusion = fixed_size_partition(
        fields, DRAW_SIZE, with_marginals=True
    )

    best = exact_next(fields[0], DRAW_SIZE)

    logp = float(
        fields[0, np.asarray(best)].sum() - log_z[0]
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

    previous_q, previous_r, _ = final_model.probabilities(
        raw[-8:], latest[-8:]
    )

    _, previous_inclusion = fixed_size_partition(
        np.log(previous_q) - np.log(previous_r),
        DRAW_SIZE,
        with_marginals=True,
    )

    sensitivity = float(
        np.max(np.abs(previous_inclusion - inclusion[0]))
    )

    reference_result = PROTOCOL.probabilities(
        np.zeros_like(final_model.alice_weights),
        np.zeros_like(final_model.bob_angles),
        final_model.transform(raw_next),
        latest_next,
        explicit=True,
    )

    reference_q, reference_r = reference_result[:2]

    reference_best = exact_next(
        (np.log(reference_q) - np.log(reference_r))[0],
        DRAW_SIZE,
    )

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
            " NEXT je maksimum njegove sačuvane validne raspodele."
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

    repetitions = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if repetitions:
        warnings.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji."
        )

    if best == tuple(int(value) for value in draws[-1]):
        warnings.append("NEXT je jednak poslednjem istorijskom kolu.")

    if best == reference_best:
        warnings.append(
            "Naučeni maksimum je isti kao referentni EARAC maksimum."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Uslovne margine slabo reaguju na proverene kontekste."
        )

    result = {
        "name": path.name,
        "hash": fingerprint,
        "next": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "probability": math.exp(logp),
        "marginal_sum": float(inclusion.sum()),
        "circuit_error": circuit_error,
        "sensitivity": sensitivity,
        "gradient": final_model.gradient_norm,
        "converged": final_model.converged,
        "warnings": warnings,
    }

    # Print each completed dataset immediately.
    print_result(result)

    return result


def print_result(result):
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
        f"Greška prema Qiskitu: {result['circuit_error']:.3e}",
        f"Osetljivost uslovnih margina: {result['sensitivity']:.9e}",
        f"Max|grad| završnog modela: {result['gradient']:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if result["converged"] else "NIJE dostignut"),
    )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    say(
        "QUANTUM POGADJANJE V2 — Qiskit EARAC — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Alice: istorijski upravljane kvantne rotacije.",
        "Bob: naučeni ortogonalni merni okvir.",
        "Trening: vektorizovana propagacija kompleksnih amplituda.",
        "Samoprovere porede amplitude sa Qiskit Statevector kolima.",
        "Završni NEXT koristi direktno Qiskit izvršavanje.",
        "Nema klasičnog izlaznog regresora.",
        "Nema slučajnog uzorkovanja, šuma ili mešanja redova.",
        "Limit optimizacije ne odbacuje validnu sačuvanu raspodelu.",
        "Konvergencija se prijavljuje odvojeno od numeričke ispravnosti.",
        f"Budžet: najviše {OPTIMIZER_STEPS} iteracija po početku.",
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

        datasets.append((path, draws, occupancy, fingerprint))
        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, occupancy, fingerprint in datasets:
        results.append(
            process_file(path, draws, occupancy, fingerprint)
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
        "NEXT je tačan maksimum raspodele za sačuvane parametre.",
        "Nepotvrđena konvergencija ostaje označena u rezultatu.",
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
            "Ranije sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM POGADJANJE V2 — Qiskit EARAC — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Alice: istorijski upravljane kvantne rotacije.
Bob: naučeni ortogonalni merni okvir.
Trening: vektorizovana propagacija kompleksnih amplituda.
Samoprovere porede amplitude sa Qiskit Statevector kolima.
Završni NEXT koristi direktno Qiskit izvršavanje.
Nema klasičnog izlaznog regresora.
Nema slučajnog uzorkovanja, šuma ili mešanja redova.
Limit optimizacije ne odbacuje validnu sačuvanu raspodelu.
Konvergencija se prijavljuje odvojeno od numeričke ispravnosti.
Budžet: najviše 600 iteracija po početku.
Kontrolne tačke: /Quantum_pogadjanje/Quantum_pogadjanje_v2_checkpoints

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
Greška amplituda prema Qiskitu: 2.238e-16
Najveća greška gradijenta: 7.198e-10
Trajanje provera: 0.0 s

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.001 | početak=17 | prelaza=2825
Iteracija 25 | proteklo 3.2 s
Najbolji cilj: 16.355939855
Max|grad| te tačke: 1.286e-02
Iteracija 50 | proteklo 6.0 s
Najbolji cilj: 16.311475917
Max|grad| te tačke: 5.258e-03
Iteracija 75 | proteklo 8.9 s
Najbolji cilj: 16.299511649
Max|grad| te tačke: 6.584e-03
Iteracija 100 | proteklo 11.7 s
Najbolji cilj: 16.296245103
Max|grad| te tačke: 2.367e-03
Iteracija 125 | proteklo 14.6 s
Najbolji cilj: 16.295043328
Max|grad| te tačke: 5.393e-03
Iteracija 150 | proteklo 17.5 s
Najbolji cilj: 16.294664771
Max|grad| te tačke: 4.946e-04
Iteracija 175 | proteklo 20.3 s
Najbolji cilj: 16.294578801
Max|grad| te tačke: 6.312e-04
Iteracija 200 | proteklo 23.0 s
Najbolji cilj: 16.294548262
Max|grad| te tačke: 1.654e-04
Iteracija 225 | proteklo 25.8 s
Najbolji cilj: 16.294545781
Max|grad| te tačke: 8.774e-05
Iteracija 250 | proteklo 28.7 s
Najbolji cilj: 16.294544987
Max|grad| te tačke: 1.155e-04
Iteracija 275 | proteklo 31.6 s
Najbolji cilj: 16.294544767
Max|grad| te tačke: 1.287e-05
Iteracija 300 | proteklo 34.4 s
Najbolji cilj: 16.294544728
Max|grad| te tačke: 8.178e-06
Iteracija 325 | proteklo 37.2 s
Najbolji cilj: 16.294544714
Max|grad| te tačke: 5.416e-06
Iteracija 350 | proteklo 40.0 s
Najbolji cilj: 16.294544711
Max|grad| te tačke: 1.888e-06
Zadržana validna tačka: 16.294544710
Max|grad|: 9.511e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 41.2 s

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.001 | početak=11 | prelaza=2825
Iteracija 25 | proteklo 3.2 s
Najbolji cilj: 16.361374623
Max|grad| te tačke: 1.776e-02
Iteracija 50 | proteklo 6.0 s
Najbolji cilj: 16.314087535
Max|grad| te tačke: 6.279e-03
Iteracija 75 | proteklo 8.8 s
Najbolji cilj: 16.300184611
Max|grad| te tačke: 9.203e-03
Iteracija 100 | proteklo 11.6 s
Najbolji cilj: 16.296505218
Max|grad| te tačke: 1.809e-03
Iteracija 125 | proteklo 14.5 s
Najbolji cilj: 16.295136662
Max|grad| te tačke: 1.992e-03
Iteracija 150 | proteklo 17.5 s
Najbolji cilj: 16.294683620
Max|grad| te tačke: 7.162e-04
Iteracija 175 | proteklo 20.4 s
Najbolji cilj: 16.294588406
Max|grad| te tačke: 4.523e-04
Iteracija 200 | proteklo 23.2 s
Najbolji cilj: 16.294548317
Max|grad| te tačke: 2.090e-04
Iteracija 225 | proteklo 26.1 s
Najbolji cilj: 16.294545778
Max|grad| te tačke: 3.447e-05
Iteracija 250 | proteklo 28.9 s
Najbolji cilj: 16.294544916
Max|grad| te tačke: 2.803e-05
Iteracija 275 | proteklo 32.0 s
Najbolji cilj: 16.294544745
Max|grad| te tačke: 1.597e-05
Iteracija 300 | proteklo 34.9 s
Najbolji cilj: 16.294544718
Max|grad| te tačke: 7.575e-06
Iteracija 325 | proteklo 37.9 s
Najbolji cilj: 16.294544712
Max|grad| te tačke: 6.774e-06
Iteracija 350 | proteklo 40.8 s
Najbolji cilj: 16.294544710
Max|grad| te tačke: 1.472e-06
Zadržana validna tačka: 16.294544710
Max|grad|: 9.037e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 42.1 s
Validacioni NLL: 16.969752472

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.001 | početak=17 | prelaza=3296
Iteracija 25 | proteklo 3.7 s
Najbolji cilj: 16.382954976
Max|grad| te tačke: 1.193e-02
Iteracija 50 | proteklo 7.0 s
Najbolji cilj: 16.359283039
Max|grad| te tačke: 5.595e-03
Iteracija 75 | proteklo 10.5 s
Najbolji cilj: 16.350281682
Max|grad| te tačke: 5.495e-03
Iteracija 100 | proteklo 13.9 s
Najbolji cilj: 16.348456609
Max|grad| te tačke: 7.713e-04
Iteracija 125 | proteklo 17.5 s
Najbolji cilj: 16.347719496
Max|grad| te tačke: 2.461e-03
Iteracija 150 | proteklo 20.8 s
Najbolji cilj: 16.346755519
Max|grad| te tačke: 7.813e-04
Iteracija 175 | proteklo 24.3 s
Najbolji cilj: 16.346696688
Max|grad| te tačke: 5.337e-04
Iteracija 200 | proteklo 28.0 s
Najbolji cilj: 16.346085359
Max|grad| te tačke: 4.246e-03
Iteracija 225 | proteklo 31.5 s
Najbolji cilj: 16.344429084
Max|grad| te tačke: 1.097e-03
Iteracija 250 | proteklo 34.9 s
Najbolji cilj: 16.344322463
Max|grad| te tačke: 4.265e-04
Iteracija 275 | proteklo 38.3 s
Najbolji cilj: 16.344289996
Max|grad| te tačke: 1.677e-04
Iteracija 300 | proteklo 41.9 s
Najbolji cilj: 16.344285797
Max|grad| te tačke: 7.149e-05
Iteracija 325 | proteklo 45.3 s
Najbolji cilj: 16.344285263
Max|grad| te tačke: 1.959e-05
Iteracija 350 | proteklo 48.8 s
Najbolji cilj: 16.344285115
Max|grad| te tačke: 2.938e-05
Iteracija 375 | proteklo 52.1 s
Najbolji cilj: 16.344285083
Max|grad| te tačke: 3.498e-06
Iteracija 400 | proteklo 55.5 s
Najbolji cilj: 16.344285078
Max|grad| te tačke: 1.603e-06
Iteracija 425 | proteklo 58.8 s
Najbolji cilj: 16.344285076
Max|grad| te tačke: 7.807e-07
Zadržana validna tačka: 16.344285076
Max|grad|: 7.807e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 59.0 s

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.001 | početak=11 | prelaza=3296
Iteracija 25 | proteklo 3.6 s
Najbolji cilj: 16.385004894
Max|grad| te tačke: 9.986e-03
Iteracija 50 | proteklo 6.8 s
Najbolji cilj: 16.359880028
Max|grad| te tačke: 6.830e-03
Iteracija 75 | proteklo 10.1 s
Najbolji cilj: 16.351063064
Max|grad| te tačke: 4.044e-03
Iteracija 100 | proteklo 13.3 s
Najbolji cilj: 16.348513273
Max|grad| te tačke: 1.011e-03
Iteracija 125 | proteklo 16.7 s
Najbolji cilj: 16.347777732
Max|grad| te tačke: 3.359e-03
Iteracija 150 | proteklo 20.1 s
Najbolji cilj: 16.346760527
Max|grad| te tačke: 4.039e-04
Iteracija 175 | proteklo 23.3 s
Najbolji cilj: 16.346704469
Max|grad| te tačke: 8.155e-04
Iteracija 200 | proteklo 27.2 s
Najbolji cilj: 16.346016618
Max|grad| te tačke: 4.686e-03
Iteracija 225 | proteklo 30.4 s
Najbolji cilj: 16.344425033
Max|grad| te tačke: 1.018e-03
Iteracija 250 | proteklo 33.8 s
Najbolji cilj: 16.344315140
Max|grad| te tačke: 2.830e-04
Iteracija 275 | proteklo 37.3 s
Najbolji cilj: 16.344291544
Max|grad| te tačke: 2.007e-04
Iteracija 300 | proteklo 40.7 s
Najbolji cilj: 16.344285683
Max|grad| te tačke: 6.395e-05
Iteracija 325 | proteklo 44.2 s
Najbolji cilj: 16.344285172
Max|grad| te tačke: 2.422e-05
Iteracija 350 | proteklo 47.5 s
Najbolji cilj: 16.344285105
Max|grad| te tačke: 2.461e-05
Iteracija 375 | proteklo 50.7 s
Najbolji cilj: 16.344285080
Max|grad| te tačke: 5.080e-06
Zadržana validna tačka: 16.344285076
Max|grad|: 9.960e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 54.2 s
Validacioni NLL: 16.858653683
KANDIDAT: lag=1 | λ=0.001
Zbirni validacioni NLL: 16.914203078
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.01 | početak=17 | prelaza=2825
Iteracija 25 | proteklo 3.1 s
Najbolji cilj: 16.570483685
Max|grad| te tačke: 6.028e-03
Iteracija 50 | proteklo 6.0 s
Najbolji cilj: 16.561494469
Max|grad| te tačke: 4.072e-03
Iteracija 75 | proteklo 8.8 s
Najbolji cilj: 16.558163473
Max|grad| te tačke: 3.585e-03
Iteracija 100 | proteklo 11.6 s
Najbolji cilj: 16.556417965
Max|grad| te tačke: 3.510e-04
Iteracija 125 | proteklo 14.5 s
Najbolji cilj: 16.556392779
Max|grad| te tačke: 1.190e-04
Iteracija 150 | proteklo 17.4 s
Najbolji cilj: 16.556392408
Max|grad| te tačke: 9.210e-06
Iteracija 175 | proteklo 20.4 s
Najbolji cilj: 16.556392400
Max|grad| te tačke: 8.257e-07
Zadržana validna tačka: 16.556392400
Max|grad|: 8.257e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 20.6 s

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.01 | početak=11 | prelaza=2825
Iteracija 25 | proteklo 3.1 s
Najbolji cilj: 16.570001339
Max|grad| te tačke: 8.865e-03
Iteracija 50 | proteklo 6.1 s
Najbolji cilj: 16.561996034
Max|grad| te tačke: 7.180e-03
Iteracija 75 | proteklo 8.9 s
Najbolji cilj: 16.557085031
Max|grad| te tačke: 2.429e-03
Iteracija 100 | proteklo 11.6 s
Najbolji cilj: 16.556404181
Max|grad| te tačke: 2.483e-04
Iteracija 125 | proteklo 14.4 s
Najbolji cilj: 16.556392566
Max|grad| te tačke: 9.192e-05
Iteracija 150 | proteklo 17.1 s
Najbolji cilj: 16.556392404
Max|grad| te tačke: 7.100e-06
Zadržana validna tačka: 16.556392400
Max|grad|: 9.407e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 19.5 s
Validacioni NLL: 16.960647058

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.01 | početak=17 | prelaza=3296
Iteracija 25 | proteklo 3.6 s
Najbolji cilj: 16.603812538
Max|grad| te tačke: 4.204e-03
Iteracija 50 | proteklo 6.8 s
Najbolji cilj: 16.599144100
Max|grad| te tačke: 1.098e-03
Iteracija 75 | proteklo 10.1 s
Najbolji cilj: 16.599038842
Max|grad| te tačke: 1.398e-04
Iteracija 100 | proteklo 13.4 s
Najbolji cilj: 16.599037134
Max|grad| te tačke: 1.434e-05
Iteracija 125 | proteklo 16.7 s
Najbolji cilj: 16.599037101
Max|grad| te tačke: 1.358e-06
Zadržana validna tačka: 16.599037101
Max|grad|: 9.171e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 17.5 s

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.01 | početak=11 | prelaza=3296
Iteracija 25 | proteklo 3.6 s
Najbolji cilj: 16.604840575
Max|grad| te tačke: 5.223e-03
Iteracija 50 | proteklo 6.8 s
Najbolji cilj: 16.599143248
Max|grad| te tačke: 1.423e-03
Iteracija 75 | proteklo 10.0 s
Najbolji cilj: 16.599038622
Max|grad| te tačke: 1.299e-04
Iteracija 100 | proteklo 13.4 s
Najbolji cilj: 16.599037134
Max|grad| te tačke: 2.068e-05
Iteracija 125 | proteklo 16.9 s
Najbolji cilj: 16.599037101
Max|grad| te tačke: 1.159e-06
Zadržana validna tačka: 16.599037101
Max|grad|: 7.239e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 18.5 s
Validacioni NLL: 16.901481435
KANDIDAT: lag=1 | λ=0.01
Zbirni validacioni NLL: 16.931064246
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.001 | početak=17 | prelaza=2825
Iteracija 25 | proteklo 3.2 s
Najbolji cilj: 15.650206474
Max|grad| te tačke: 5.260e-02
Iteracija 50 | proteklo 6.0 s
Najbolji cilj: 15.426028659
Max|grad| te tačke: 1.831e-01
Iteracija 75 | proteklo 8.9 s
Najbolji cilj: 15.194874456
Max|grad| te tačke: 3.736e-02
Iteracija 100 | proteklo 11.8 s
Najbolji cilj: 15.070433615
Max|grad| te tačke: 4.164e-02
Iteracija 125 | proteklo 14.6 s
Najbolji cilj: 15.015350132
Max|grad| te tačke: 2.636e-02
Iteracija 150 | proteklo 17.5 s
Najbolji cilj: 14.986859763
Max|grad| te tačke: 2.280e-02
Iteracija 175 | proteklo 20.3 s
Najbolji cilj: 14.965082313
Max|grad| te tačke: 2.386e-02
Iteracija 200 | proteklo 23.2 s
Najbolji cilj: 14.952656015
Max|grad| te tačke: 1.118e-02
Iteracija 225 | proteklo 26.0 s
Najbolji cilj: 14.944737336
Max|grad| te tačke: 4.570e-03
Iteracija 250 | proteklo 29.0 s
Najbolji cilj: 14.938949109
Max|grad| te tačke: 5.542e-03
Iteracija 275 | proteklo 31.9 s
Najbolji cilj: 14.934265238
Max|grad| te tačke: 6.335e-03
Iteracija 300 | proteklo 34.8 s
Najbolji cilj: 14.933395650
Max|grad| te tačke: 1.518e-03
Iteracija 325 | proteklo 38.0 s
Najbolji cilj: 14.932371734
Max|grad| te tačke: 1.428e-03
Iteracija 350 | proteklo 41.0 s
Najbolji cilj: 14.931940830
Max|grad| te tačke: 1.031e-03
Iteracija 375 | proteklo 43.9 s
Najbolji cilj: 14.931600147
Max|grad| te tačke: 6.183e-04
Iteracija 400 | proteklo 47.0 s
Najbolji cilj: 14.931028933
Max|grad| te tačke: 4.224e-03
Iteracija 425 | proteklo 50.4 s
Najbolji cilj: 14.927351346
Max|grad| te tačke: 2.457e-03
Iteracija 450 | proteklo 53.6 s
Najbolji cilj: 14.926295127
Max|grad| te tačke: 9.360e-03
Iteracija 475 | proteklo 56.6 s
Najbolji cilj: 14.925740194
Max|grad| te tačke: 8.694e-04
Iteracija 500 | proteklo 59.4 s
Najbolji cilj: 14.925604158
Max|grad| te tačke: 1.457e-03
Iteracija 525 | proteklo 62.5 s
Najbolji cilj: 14.925592511
Max|grad| te tačke: 1.488e-04
Iteracija 550 | proteklo 65.7 s
Najbolji cilj: 14.925587954
Max|grad| te tačke: 7.215e-04
Iteracija 575 | proteklo 68.8 s
Najbolji cilj: 14.924974728
Max|grad| te tačke: 5.142e-03
Iteracija 600 | proteklo 71.7 s
Najbolji cilj: 14.923947525
Max|grad| te tačke: 2.928e-03
Zadržana validna tačka: 14.923947525
Max|grad|: 2.928e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 71.9 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.001 | početak=11 | prelaza=2825
Iteracija 25 | proteklo 3.4 s
Najbolji cilj: 15.661815828
Max|grad| te tačke: 1.936e-01
Iteracija 50 | proteklo 6.7 s
Najbolji cilj: 15.485265057
Max|grad| te tačke: 3.593e-02
Iteracija 75 | proteklo 9.6 s
Najbolji cilj: 15.273915385
Max|grad| te tačke: 5.056e-02
Iteracija 100 | proteklo 12.5 s
Najbolji cilj: 15.132276117
Max|grad| te tačke: 1.358e-02
Iteracija 125 | proteklo 15.4 s
Najbolji cilj: 15.048494602
Max|grad| te tačke: 3.658e-02
Iteracija 150 | proteklo 18.4 s
Najbolji cilj: 15.012717856
Max|grad| te tačke: 1.429e-02
Iteracija 175 | proteklo 21.4 s
Najbolji cilj: 14.980253659
Max|grad| te tačke: 1.077e-02
Iteracija 200 | proteklo 24.3 s
Najbolji cilj: 14.945534818
Max|grad| te tačke: 1.511e-02
Iteracija 225 | proteklo 27.2 s
Najbolji cilj: 14.929651324
Max|grad| te tačke: 6.604e-03
Iteracija 250 | proteklo 30.2 s
Najbolji cilj: 14.923501697
Max|grad| te tačke: 8.724e-03
Iteracija 275 | proteklo 33.1 s
Najbolji cilj: 14.918673289
Max|grad| te tačke: 6.502e-03
Iteracija 300 | proteklo 36.0 s
Najbolji cilj: 14.905939832
Max|grad| te tačke: 6.161e-03
Iteracija 325 | proteklo 38.8 s
Najbolji cilj: 14.896066646
Max|grad| te tačke: 1.523e-02
Iteracija 350 | proteklo 41.7 s
Najbolji cilj: 14.890002310
Max|grad| te tačke: 4.015e-03
Iteracija 375 | proteklo 44.6 s
Najbolji cilj: 14.887357837
Max|grad| te tačke: 1.540e-03
Iteracija 400 | proteklo 47.6 s
Najbolji cilj: 14.884107220
Max|grad| te tačke: 7.942e-03
Iteracija 425 | proteklo 50.6 s
Najbolji cilj: 14.879155754
Max|grad| te tačke: 5.140e-03
Iteracija 450 | proteklo 53.4 s
Najbolji cilj: 14.874508832
Max|grad| te tačke: 1.121e-02
Iteracija 475 | proteklo 56.2 s
Najbolji cilj: 14.872435354
Max|grad| te tačke: 9.534e-03
Iteracija 500 | proteklo 59.1 s
Najbolji cilj: 14.867908303
Max|grad| te tačke: 5.743e-03
Iteracija 525 | proteklo 62.0 s
Najbolji cilj: 14.865924788
Max|grad| te tačke: 2.424e-03
Iteracija 550 | proteklo 64.9 s
Najbolji cilj: 14.865727976
Max|grad| te tačke: 1.677e-03
Iteracija 575 | proteklo 67.9 s
Najbolji cilj: 14.864449746
Max|grad| te tačke: 4.593e-03
Iteracija 600 | proteklo 70.7 s
Najbolji cilj: 14.863301877
Max|grad| te tačke: 2.208e-03
Zadržana validna tačka: 14.863301877
Max|grad|: 2.208e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 71.0 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.
Validacioni NLL: 20.623748510

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.001 | početak=17 | prelaza=3296
Iteracija 25 | proteklo 3.9 s
Najbolji cilj: 15.880466954
Max|grad| te tačke: 3.790e-02
Iteracija 50 | proteklo 7.3 s
Najbolji cilj: 15.765185624
Max|grad| te tačke: 1.580e-02
Iteracija 75 | proteklo 10.8 s
Najbolji cilj: 15.615430418
Max|grad| te tačke: 5.505e-02
Iteracija 100 | proteklo 14.1 s
Najbolji cilj: 15.502475759
Max|grad| te tačke: 2.120e-02
Iteracija 125 | proteklo 17.4 s
Najbolji cilj: 15.432376172
Max|grad| te tačke: 1.474e-02
Iteracija 150 | proteklo 20.7 s
Najbolji cilj: 15.374165177
Max|grad| te tačke: 3.144e-02
Iteracija 175 | proteklo 24.0 s
Najbolji cilj: 15.340581112
Max|grad| te tačke: 7.868e-03
Iteracija 200 | proteklo 27.4 s
Najbolji cilj: 15.313425719
Max|grad| te tačke: 1.331e-02
Iteracija 225 | proteklo 30.7 s
Najbolji cilj: 15.297106345
Max|grad| te tačke: 1.017e-02
Iteracija 250 | proteklo 34.0 s
Najbolji cilj: 15.278116387
Max|grad| te tačke: 2.700e-02
Iteracija 275 | proteklo 37.3 s
Najbolji cilj: 15.271772370
Max|grad| te tačke: 2.317e-03
Iteracija 300 | proteklo 40.9 s
Najbolji cilj: 15.267918405
Max|grad| te tačke: 2.932e-03
Iteracija 325 | proteklo 44.5 s
Najbolji cilj: 15.264670471
Max|grad| te tačke: 5.676e-03
Iteracija 350 | proteklo 47.9 s
Najbolji cilj: 15.261368037
Max|grad| te tačke: 6.748e-03
Iteracija 375 | proteklo 51.2 s
Najbolji cilj: 15.260849186
Max|grad| te tačke: 1.934e-03
Iteracija 400 | proteklo 54.5 s
Najbolji cilj: 15.260528055
Max|grad| te tačke: 1.560e-03
Iteracija 425 | proteklo 58.0 s
Najbolji cilj: 15.260377598
Max|grad| te tačke: 1.170e-03
Iteracija 450 | proteklo 61.5 s
Najbolji cilj: 15.260243079
Max|grad| te tačke: 6.065e-04
Iteracija 475 | proteklo 64.9 s
Najbolji cilj: 15.260220992
Max|grad| te tačke: 4.966e-04
Iteracija 500 | proteklo 68.4 s
Najbolji cilj: 15.260210422
Max|grad| te tačke: 5.843e-04
Iteracija 525 | proteklo 72.2 s
Najbolji cilj: 15.260192583
Max|grad| te tačke: 1.776e-03
Iteracija 550 | proteklo 75.8 s
Najbolji cilj: 15.260177288
Max|grad| te tačke: 3.668e-04
Iteracija 575 | proteklo 79.3 s
Najbolji cilj: 15.260174400
Max|grad| te tačke: 3.158e-04
Iteracija 600 | proteklo 82.9 s
Najbolji cilj: 15.260169812
Max|grad| te tačke: 3.778e-04
Zadržana validna tačka: 15.260169812
Max|grad|: 3.778e-04
Gradijentni uslov: dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 83.2 s

...

Iteracija 25 | proteklo 1.2 s
Najbolji cilj: 15.908631056
Max|grad| te tačke: 1.074e-02
Iteracija 50 | proteklo 2.4 s
Najbolji cilj: 15.851492401
Max|grad| te tačke: 9.376e-03
Iteracija 75 | proteklo 3.5 s
Najbolji cilj: 15.820116244
Max|grad| te tačke: 1.498e-02
Iteracija 100 | proteklo 4.7 s
Najbolji cilj: 15.787096667
Max|grad| te tačke: 1.495e-02
Iteracija 125 | proteklo 5.8 s
Najbolji cilj: 15.768345836
Max|grad| te tačke: 2.139e-03
Iteracija 150 | proteklo 6.8 s
Najbolji cilj: 15.766169318
Max|grad| te tačke: 8.057e-04
Iteracija 175 | proteklo 8.0 s
Najbolji cilj: 15.762350957
Max|grad| te tačke: 1.626e-03
Iteracija 200 | proteklo 9.2 s
Najbolji cilj: 15.760562946
Max|grad| te tačke: 7.752e-04
Iteracija 225 | proteklo 10.3 s
Najbolji cilj: 15.760531120
Max|grad| te tačke: 3.689e-05
Iteracija 250 | proteklo 11.4 s
Najbolji cilj: 15.760531010
Max|grad| te tačke: 2.736e-06
Zadržana validna tačka: 15.760531010
Max|grad|: 9.277e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 11.7 s

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.01 | početak=11 | prelaza=1039
Iteracija 25 | proteklo 1.2 s
Najbolji cilj: 15.911120089
Max|grad| te tačke: 9.639e-03
Iteracija 50 | proteklo 2.4 s
Najbolji cilj: 15.858939276
Max|grad| te tačke: 7.196e-03
Iteracija 75 | proteklo 3.5 s
Najbolji cilj: 15.827097459
Max|grad| te tačke: 6.345e-03
Iteracija 100 | proteklo 4.7 s
Najbolji cilj: 15.806941538
Max|grad| te tačke: 5.967e-03
Iteracija 125 | proteklo 5.8 s
Najbolji cilj: 15.772945081
Max|grad| te tačke: 7.624e-03
Iteracija 150 | proteklo 7.0 s
Najbolji cilj: 15.767665891
Max|grad| te tačke: 3.501e-04
Iteracija 175 | proteklo 8.2 s
Najbolji cilj: 15.766867780
Max|grad| te tačke: 5.157e-03
Iteracija 200 | proteklo 9.4 s
Najbolji cilj: 15.764075148
Max|grad| te tačke: 1.810e-04
Iteracija 225 | proteklo 10.5 s
Najbolji cilj: 15.764070661
Max|grad| te tačke: 1.584e-05
Iteracija 250 | proteklo 11.7 s
Najbolji cilj: 15.764070634
Max|grad| te tačke: 1.122e-06
Zadržana validna tačka: 15.764070634
Max|grad|: 9.014e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 12.0 s
Validacioni NLL: 19.204521641

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.01 | početak=17 | prelaza=1212
Iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.092334225
Max|grad| te tačke: 1.900e-02
Iteracija 50 | proteklo 2.7 s
Najbolji cilj: 16.057235653
Max|grad| te tačke: 6.461e-03
Iteracija 75 | proteklo 4.0 s
Najbolji cilj: 16.039971645
Max|grad| te tačke: 5.399e-03
Iteracija 100 | proteklo 5.2 s
Najbolji cilj: 16.033533875
Max|grad| te tačke: 1.247e-03
Iteracija 125 | proteklo 6.7 s
Najbolji cilj: 16.029526308
Max|grad| te tačke: 1.112e-03
Iteracija 150 | proteklo 8.1 s
Najbolji cilj: 16.027929328
Max|grad| te tačke: 4.040e-03
Iteracija 175 | proteklo 9.5 s
Najbolji cilj: 16.011151484
Max|grad| te tačke: 3.745e-03
Iteracija 200 | proteklo 10.8 s
Najbolji cilj: 16.004300765
Max|grad| te tačke: 5.340e-04
Iteracija 225 | proteklo 12.1 s
Najbolji cilj: 16.004278396
Max|grad| te tačke: 4.725e-05
Iteracija 250 | proteklo 13.4 s
Najbolji cilj: 16.004278014
Max|grad| te tačke: 1.063e-05
Iteracija 275 | proteklo 14.6 s
Najbolji cilj: 16.004277991
Max|grad| te tačke: 2.906e-06
Zadržana validna tačka: 16.004277991
Max|grad|: 9.939e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 15.0 s

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.01 | početak=11 | prelaza=1212
Iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.095491114
Max|grad| te tačke: 1.246e-02
Iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.060241295
Max|grad| te tačke: 7.154e-03
Iteracija 75 | proteklo 3.9 s
Najbolji cilj: 16.039339750
Max|grad| te tačke: 8.048e-03
Iteracija 100 | proteklo 5.1 s
Najbolji cilj: 16.033535584
Max|grad| te tačke: 1.166e-03
Iteracija 125 | proteklo 6.5 s
Najbolji cilj: 16.029594254
Max|grad| te tačke: 1.372e-03
Iteracija 150 | proteklo 7.8 s
Najbolji cilj: 16.021629713
Max|grad| te tačke: 1.580e-02
Iteracija 175 | proteklo 9.0 s
Najbolji cilj: 16.007356075
Max|grad| te tačke: 4.596e-03
Iteracija 200 | proteklo 10.3 s
Najbolji cilj: 16.004297338
Max|grad| te tačke: 2.725e-04
Iteracija 225 | proteklo 11.5 s
Najbolji cilj: 16.004279437
Max|grad| te tačke: 1.290e-04
Iteracija 250 | proteklo 12.8 s
Najbolji cilj: 16.004278013
Max|grad| te tačke: 1.781e-05
Zadržana validna tačka: 16.004277991
Max|grad|: 9.582e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 13.9 s
Validacioni NLL: 18.632486136
KANDIDAT: lag=1 | λ=0.01
Zbirni validacioni NLL: 18.918503889
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.001 | početak=17 | prelaza=1039
Iteracija 25 | proteklo 1.2 s
Najbolji cilj: 11.750088854
Max|grad| te tačke: 8.292e-02
Iteracija 50 | proteklo 2.4 s
Najbolji cilj: 10.645615560
Max|grad| te tačke: 6.535e-02
Iteracija 75 | proteklo 3.5 s
Najbolji cilj: 10.076658209
Max|grad| te tačke: 5.239e-02
Iteracija 100 | proteklo 4.7 s
Najbolji cilj: 9.800541778
Max|grad| te tačke: 3.461e-02
Iteracija 125 | proteklo 5.8 s
Najbolji cilj: 9.657761136
Max|grad| te tačke: 2.422e-02
Iteracija 150 | proteklo 6.9 s
Najbolji cilj: 9.560448063
Max|grad| te tačke: 2.501e-02
Iteracija 175 | proteklo 8.0 s
Najbolji cilj: 9.502025368
Max|grad| te tačke: 2.143e-02
Iteracija 200 | proteklo 9.2 s
Najbolji cilj: 9.444334372
Max|grad| te tačke: 1.788e-02
Iteracija 225 | proteklo 10.3 s
Najbolji cilj: 9.403652773
Max|grad| te tačke: 9.029e-03
Iteracija 250 | proteklo 11.4 s
Najbolji cilj: 9.370033695
Max|grad| te tačke: 9.201e-03
Iteracija 275 | proteklo 12.6 s
Najbolji cilj: 9.347324345
Max|grad| te tačke: 2.180e-02
Iteracija 300 | proteklo 13.7 s
Najbolji cilj: 9.321124080
Max|grad| te tačke: 2.181e-02
Iteracija 325 | proteklo 14.8 s
Najbolji cilj: 9.307843189
Max|grad| te tačke: 8.358e-03
Iteracija 350 | proteklo 15.9 s
Najbolji cilj: 9.293027527
Max|grad| te tačke: 2.854e-03
Iteracija 375 | proteklo 17.1 s
Najbolji cilj: 9.284873189
Max|grad| te tačke: 5.919e-03
Iteracija 400 | proteklo 18.2 s
Najbolji cilj: 9.271202699
Max|grad| te tačke: 4.464e-03
Iteracija 425 | proteklo 19.3 s
Najbolji cilj: 9.258159513
Max|grad| te tačke: 6.436e-03
Iteracija 450 | proteklo 20.5 s
Najbolji cilj: 9.250631859
Max|grad| te tačke: 8.038e-03
Iteracija 475 | proteklo 21.6 s
Najbolji cilj: 9.244341221
Max|grad| te tačke: 1.812e-03
Iteracija 500 | proteklo 22.8 s
Najbolji cilj: 9.238060328
Max|grad| te tačke: 5.142e-03
Iteracija 525 | proteklo 23.9 s
Najbolji cilj: 9.230845850
Max|grad| te tačke: 5.988e-03
Iteracija 550 | proteklo 25.1 s
Najbolji cilj: 9.222132605
Max|grad| te tačke: 6.736e-03
Iteracija 575 | proteklo 26.2 s
Najbolji cilj: 9.219703138
Max|grad| te tačke: 7.133e-03
Iteracija 600 | proteklo 27.3 s
Najbolji cilj: 9.218233114
Max|grad| te tačke: 4.166e-03
Zadržana validna tačka: 9.218233114
Max|grad|: 4.166e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 27.4 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.001 | početak=11 | prelaza=1039
Iteracija 25 | proteklo 1.2 s
Najbolji cilj: 11.857776131
Max|grad| te tačke: 4.655e-01
Iteracija 50 | proteklo 2.5 s
Najbolji cilj: 10.711850743
Max|grad| te tačke: 6.857e-02
Iteracija 75 | proteklo 3.6 s
Najbolji cilj: 10.155116068
Max|grad| te tačke: 2.822e-02
Iteracija 100 | proteklo 4.8 s
Najbolji cilj: 9.863679911
Max|grad| te tačke: 2.281e-02
Iteracija 125 | proteklo 5.9 s
Najbolji cilj: 9.686955471
Max|grad| te tačke: 3.183e-02
Iteracija 150 | proteklo 7.0 s
Najbolji cilj: 9.570823557
Max|grad| te tačke: 3.443e-02
Iteracija 175 | proteklo 8.1 s
Najbolji cilj: 9.487984408
Max|grad| te tačke: 3.162e-02
Iteracija 200 | proteklo 9.3 s
Najbolji cilj: 9.442009283
Max|grad| te tačke: 2.258e-02
Iteracija 225 | proteklo 10.5 s
Najbolji cilj: 9.416931607
Max|grad| te tačke: 7.743e-03
Iteracija 250 | proteklo 11.6 s
Najbolji cilj: 9.394415003
Max|grad| te tačke: 1.907e-02
Iteracija 275 | proteklo 12.8 s
Najbolji cilj: 9.369882194
Max|grad| te tačke: 1.147e-02
Iteracija 300 | proteklo 13.9 s
Najbolji cilj: 9.337222613
Max|grad| te tačke: 2.551e-02
Iteracija 325 | proteklo 15.1 s
Najbolji cilj: 9.305263150
Max|grad| te tačke: 1.160e-02
Iteracija 350 | proteklo 16.2 s
Najbolji cilj: 9.289242964
Max|grad| te tačke: 8.404e-03
Iteracija 375 | proteklo 17.3 s
Najbolji cilj: 9.273722103
Max|grad| te tačke: 1.598e-02
Iteracija 400 | proteklo 18.5 s
Najbolji cilj: 9.256402321
Max|grad| te tačke: 1.881e-02
Iteracija 425 | proteklo 19.7 s
Najbolji cilj: 9.243224470
Max|grad| te tačke: 1.047e-02
Iteracija 450 | proteklo 20.8 s
Najbolji cilj: 9.232835298
Max|grad| te tačke: 3.985e-03
Iteracija 475 | proteklo 21.9 s
Najbolji cilj: 9.228664992
Max|grad| te tačke: 2.828e-03
Iteracija 500 | proteklo 23.2 s
Najbolji cilj: 9.224796975
Max|grad| te tačke: 1.107e-02
Iteracija 525 | proteklo 24.3 s
Najbolji cilj: 9.217007756
Max|grad| te tačke: 7.710e-03
Iteracija 550 | proteklo 25.5 s
Najbolji cilj: 9.208537146
Max|grad| te tačke: 5.876e-03
Iteracija 575 | proteklo 26.6 s
Najbolji cilj: 9.197900920
Max|grad| te tačke: 3.971e-03
Iteracija 600 | proteklo 27.8 s
Najbolji cilj: 9.190050348
Max|grad| te tačke: 5.215e-03
Zadržana validna tačka: 9.190050348
Max|grad|: 5.215e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 27.9 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.
Validacioni NLL: 23.851852077

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.001 | početak=17 | prelaza=1212
Iteracija 25 | proteklo 1.5 s
Najbolji cilj: 12.672338598
Max|grad| te tačke: 7.842e-02
Iteracija 50 | proteklo 2.9 s
Najbolji cilj: 11.722316274
Max|grad| te tačke: 9.468e-02
Iteracija 75 | proteklo 4.2 s
Najbolji cilj: 11.121693295
Max|grad| te tačke: 3.319e-02
Iteracija 100 | proteklo 5.5 s
Najbolji cilj: 10.896807899
Max|grad| te tačke: 3.962e-02
Iteracija 125 | proteklo 6.7 s
Najbolji cilj: 10.733891662
Max|grad| te tačke: 3.029e-02
Iteracija 150 | proteklo 8.0 s
Najbolji cilj: 10.612754573
Max|grad| te tačke: 3.978e-02
Iteracija 175 | proteklo 9.3 s
Najbolji cilj: 10.506632988
Max|grad| te tačke: 1.900e-02
Iteracija 200 | proteklo 10.6 s
Najbolji cilj: 10.431165339
Max|grad| te tačke: 2.405e-02
Iteracija 225 | proteklo 11.9 s
Najbolji cilj: 10.371491254
Max|grad| te tačke: 2.271e-02
Iteracija 250 | proteklo 13.1 s
Najbolji cilj: 10.320195978
Max|grad| te tačke: 2.992e-02
Iteracija 275 | proteklo 14.4 s
Najbolji cilj: 10.270502097
Max|grad| te tačke: 1.668e-02
Iteracija 300 | proteklo 15.7 s
Najbolji cilj: 10.245216261
Max|grad| te tačke: 2.804e-02
Iteracija 325 | proteklo 17.0 s
Najbolji cilj: 10.233423562
Max|grad| te tačke: 5.552e-03
Iteracija 350 | proteklo 18.3 s
Najbolji cilj: 10.220367533
Max|grad| te tačke: 1.268e-02
Iteracija 375 | proteklo 19.6 s
Najbolji cilj: 10.214108374
Max|grad| te tačke: 9.738e-03
Iteracija 400 | proteklo 20.9 s
Najbolji cilj: 10.202549675
Max|grad| te tačke: 1.870e-02
Iteracija 425 | proteklo 22.2 s
Najbolji cilj: 10.194626897
Max|grad| te tačke: 3.154e-03
Iteracija 450 | proteklo 23.5 s
Najbolji cilj: 10.183340202
Max|grad| te tačke: 4.117e-03
Iteracija 475 | proteklo 24.9 s
Najbolji cilj: 10.177854536
Max|grad| te tačke: 3.879e-03
Iteracija 500 | proteklo 26.3 s
Najbolji cilj: 10.170251030
Max|grad| te tačke: 1.249e-02
Iteracija 525 | proteklo 27.5 s
Najbolji cilj: 10.168176348
Max|grad| te tačke: 3.695e-03
Iteracija 550 | proteklo 28.9 s
Najbolji cilj: 10.166293472
Max|grad| te tačke: 4.308e-03
Iteracija 575 | proteklo 30.3 s
Najbolji cilj: 10.165614748
Max|grad| te tačke: 1.453e-02
Iteracija 600 | proteklo 31.6 s
Najbolji cilj: 10.162968261
Max|grad| te tačke: 4.744e-03
Zadržana validna tačka: 10.162968261
Max|grad|: 4.744e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 31.7 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.001 | početak=11 | prelaza=1212
Iteracija 25 | proteklo 1.4 s
Najbolji cilj: 12.536962859
Max|grad| te tačke: 1.367e-01
Iteracija 50 | proteklo 2.6 s
Najbolji cilj: 11.482120085
Max|grad| te tačke: 5.144e-02
Iteracija 75 | proteklo 4.0 s
Najbolji cilj: 10.984708292
Max|grad| te tačke: 3.722e-02
Iteracija 100 | proteklo 5.2 s
Najbolji cilj: 10.739687765
Max|grad| te tačke: 3.508e-02
Iteracija 125 | proteklo 6.5 s
Najbolji cilj: 10.572113767
Max|grad| te tačke: 4.773e-02
Iteracija 150 | proteklo 7.8 s
Najbolji cilj: 10.470367674
Max|grad| te tačke: 1.313e-02
Iteracija 175 | proteklo 9.1 s
Najbolji cilj: 10.397593650
Max|grad| te tačke: 2.612e-02
Iteracija 200 | proteklo 10.4 s
Najbolji cilj: 10.338729275
Max|grad| te tačke: 1.855e-02
Iteracija 225 | proteklo 11.6 s
Najbolji cilj: 10.312048218
Max|grad| te tačke: 2.080e-02
Iteracija 250 | proteklo 12.9 s
Najbolji cilj: 10.290264279
Max|grad| te tačke: 1.821e-02
Iteracija 275 | proteklo 14.2 s
Najbolji cilj: 10.272164538
Max|grad| te tačke: 1.469e-02
Iteracija 300 | proteklo 15.5 s
Najbolji cilj: 10.248706835
Max|grad| te tačke: 1.072e-02
Iteracija 325 | proteklo 16.7 s
Najbolji cilj: 10.233859340
Max|grad| te tačke: 1.450e-02
Iteracija 350 | proteklo 18.1 s
Najbolji cilj: 10.210615175
Max|grad| te tačke: 1.760e-02
Iteracija 375 | proteklo 19.4 s
Najbolji cilj: 10.194236840
Max|grad| te tačke: 2.456e-02
Iteracija 400 | proteklo 20.7 s
Najbolji cilj: 10.184060844
Max|grad| te tačke: 1.717e-02
Iteracija 425 | proteklo 21.9 s
Najbolji cilj: 10.178058137
Max|grad| te tačke: 4.717e-03
Iteracija 450 | proteklo 23.3 s
Najbolji cilj: 10.176213428
Max|grad| te tačke: 6.117e-03
Iteracija 475 | proteklo 24.6 s
Najbolji cilj: 10.171953952
Max|grad| te tačke: 2.466e-03
Iteracija 500 | proteklo 25.9 s
Najbolji cilj: 10.169808802
Max|grad| te tačke: 5.126e-03
Iteracija 525 | proteklo 27.2 s
Najbolji cilj: 10.166228427
Max|grad| te tačke: 1.495e-02
Iteracija 550 | proteklo 28.5 s
Najbolji cilj: 10.161156111
Max|grad| te tačke: 1.016e-02
Iteracija 575 | proteklo 29.7 s
Najbolji cilj: 10.152709060
Max|grad| te tačke: 5.128e-03
Iteracija 600 | proteklo 31.0 s
Najbolji cilj: 10.149685867
Max|grad| te tačke: 6.560e-03
Zadržana validna tačka: 10.149685867
Max|grad|: 6.560e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 31.1 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.
Validacioni NLL: 24.433697997
KANDIDAT: lag=3 | λ=0.001
Zbirni validacioni NLL: 24.142775037
Podela bez potvrđenog gradijentnog uslova: 2

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.01 | početak=17 | prelaza=1039
Iteracija 25 | proteklo 1.2 s
Najbolji cilj: 12.377469968
Max|grad| te tačke: 4.529e-01
Iteracija 50 | proteklo 2.4 s
Najbolji cilj: 11.693163375
Max|grad| te tačke: 4.009e-02
Iteracija 75 | proteklo 3.5 s
Najbolji cilj: 11.347586674
Max|grad| te tačke: 4.000e-02
Iteracija 100 | proteklo 4.6 s
Najbolji cilj: 11.133858695
Max|grad| te tačke: 4.565e-02
Iteracija 125 | proteklo 5.8 s
Najbolji cilj: 10.981874336
Max|grad| te tačke: 5.193e-02
Iteracija 150 | proteklo 6.9 s
Najbolji cilj: 10.917306533
Max|grad| te tačke: 2.460e-02
Iteracija 175 | proteklo 8.0 s
Najbolji cilj: 10.861327294
Max|grad| te tačke: 3.726e-02
Iteracija 200 | proteklo 9.2 s
Najbolji cilj: 10.811730047
Max|grad| te tačke: 5.069e-02
Iteracija 225 | proteklo 10.3 s
Najbolji cilj: 10.780505784
Max|grad| te tačke: 1.557e-02
Iteracija 250 | proteklo 11.4 s
Najbolji cilj: 10.761782101
Max|grad| te tačke: 1.033e-02
Iteracija 275 | proteklo 12.6 s
Najbolji cilj: 10.749285721
Max|grad| te tačke: 1.001e-02
Iteracija 300 | proteklo 13.8 s
Najbolji cilj: 10.738664220
Max|grad| te tačke: 8.403e-03
Iteracija 325 | proteklo 14.9 s
Najbolji cilj: 10.720145955
Max|grad| te tačke: 1.112e-02
Iteracija 350 | proteklo 16.1 s
Najbolji cilj: 10.706484126
Max|grad| te tačke: 1.746e-02
Iteracija 375 | proteklo 17.2 s
Najbolji cilj: 10.692871262
Max|grad| te tačke: 4.275e-03
Iteracija 400 | proteklo 18.3 s
Najbolji cilj: 10.687758785
Max|grad| te tačke: 3.216e-03
Iteracija 425 | proteklo 19.5 s
Najbolji cilj: 10.686126104
Max|grad| te tačke: 2.742e-03
Iteracija 450 | proteklo 20.6 s
Najbolji cilj: 10.681519749
Max|grad| te tačke: 3.540e-03
Iteracija 475 | proteklo 21.8 s
Najbolji cilj: 10.678321806
Max|grad| te tačke: 2.433e-03
Iteracija 500 | proteklo 22.9 s
Najbolji cilj: 10.676853123
Max|grad| te tačke: 1.603e-03
Iteracija 525 | proteklo 24.0 s
Najbolji cilj: 10.676241337
Max|grad| te tačke: 7.390e-03
Iteracija 550 | proteklo 25.1 s
Najbolji cilj: 10.671680595
Max|grad| te tačke: 1.295e-02
Iteracija 575 | proteklo 26.3 s
Najbolji cilj: 10.669440132
Max|grad| te tačke: 5.899e-03
Iteracija 600 | proteklo 27.4 s
Najbolji cilj: 10.668297277
Max|grad| te tačke: 4.726e-04
Zadržana validna tačka: 10.668297277
Max|grad|: 4.726e-04
Gradijentni uslov: dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 27.5 s

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.01 | početak=11 | prelaza=1039
Iteracija 25 | proteklo 1.3 s
Najbolji cilj: 12.419821506
Max|grad| te tačke: 9.147e-02
Iteracija 50 | proteklo 2.4 s
Najbolji cilj: 11.509004750
Max|grad| te tačke: 5.529e-02
Iteracija 75 | proteklo 3.5 s
Najbolji cilj: 11.239425535
Max|grad| te tačke: 1.899e-02
Iteracija 100 | proteklo 4.6 s
Najbolji cilj: 11.082903383
Max|grad| te tačke: 5.033e-02
Iteracija 125 | proteklo 5.8 s
Najbolji cilj: 10.974442554
Max|grad| te tačke: 2.262e-02
Iteracija 150 | proteklo 6.9 s
Najbolji cilj: 10.879268833
Max|grad| te tačke: 2.958e-02
Iteracija 175 | proteklo 8.1 s
Najbolji cilj: 10.823833256
Max|grad| te tačke: 2.290e-02
Iteracija 200 | proteklo 9.2 s
Najbolji cilj: 10.788446932
Max|grad| te tačke: 2.758e-02
Iteracija 225 | proteklo 10.3 s
Najbolji cilj: 10.759890491
Max|grad| te tačke: 2.449e-02
Iteracija 250 | proteklo 11.4 s
Najbolji cilj: 10.746943128
Max|grad| te tačke: 8.390e-03
Iteracija 275 | proteklo 12.5 s
Najbolji cilj: 10.733580408
Max|grad| te tačke: 1.273e-02
Iteracija 300 | proteklo 13.7 s
Najbolji cilj: 10.724549478
Max|grad| te tačke: 5.554e-03
Iteracija 325 | proteklo 14.9 s
Najbolji cilj: 10.720261841
Max|grad| te tačke: 4.495e-03
Iteracija 350 | proteklo 16.1 s
Najbolji cilj: 10.716201218
Max|grad| te tačke: 1.323e-02
Iteracija 375 | proteklo 17.2 s
Najbolji cilj: 10.715071142
Max|grad| te tačke: 2.090e-03
Iteracija 400 | proteklo 18.4 s
Najbolji cilj: 10.714805376
Max|grad| te tačke: 1.179e-04
Iteracija 425 | proteklo 19.8 s
Najbolji cilj: 10.714800669
Max|grad| te tačke: 8.555e-04
Iteracija 450 | proteklo 21.1 s
Najbolji cilj: 10.714643288
Max|grad| te tačke: 1.119e-03
Iteracija 475 | proteklo 22.4 s
Najbolji cilj: 10.714472950
Max|grad| te tačke: 1.328e-04
Iteracija 500 | proteklo 23.6 s
Najbolji cilj: 10.714209066
Max|grad| te tačke: 3.916e-04
Iteracija 525 | proteklo 24.8 s
Najbolji cilj: 10.714193542
Max|grad| te tačke: 8.944e-05
Iteracija 550 | proteklo 25.9 s
Najbolji cilj: 10.714193262
Max|grad| te tačke: 1.465e-05
Iteracija 575 | proteklo 27.1 s
Najbolji cilj: 10.714193254
Max|grad| te tačke: 1.993e-06
Zadržana validna tačka: 10.714193254
Max|grad|: 6.783e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 27.7 s
Validacioni NLL: 23.896706991

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.01 | početak=17 | prelaza=1212
Iteracija 25 | proteklo 1.4 s
Najbolji cilj: 13.273345317
Max|grad| te tačke: 7.419e-02
Iteracija 50 | proteklo 2.6 s
Najbolji cilj: 12.526746052
Max|grad| te tačke: 4.233e-02
Iteracija 75 | proteklo 4.0 s
Najbolji cilj: 12.169118033
Max|grad| te tačke: 2.662e-02
Iteracija 100 | proteklo 5.2 s
Najbolji cilj: 11.971072019
Max|grad| te tačke: 4.368e-02
Iteracija 125 | proteklo 6.5 s
Najbolji cilj: 11.842313247
Max|grad| te tačke: 3.561e-02
Iteracija 150 | proteklo 7.8 s
Najbolji cilj: 11.776341503
Max|grad| te tačke: 1.950e-02
Iteracija 175 | proteklo 9.0 s
Najbolji cilj: 11.708511041
Max|grad| te tačke: 3.037e-02
Iteracija 200 | proteklo 10.3 s
Najbolji cilj: 11.668305388
Max|grad| te tačke: 4.366e-02
Iteracija 225 | proteklo 11.6 s
Najbolji cilj: 11.642234750
Max|grad| te tačke: 3.794e-02
Iteracija 250 | proteklo 12.8 s
Najbolji cilj: 11.625313339
Max|grad| te tačke: 6.710e-03
Iteracija 275 | proteklo 14.1 s
Najbolji cilj: 11.600630463
Max|grad| te tačke: 2.059e-02
Iteracija 300 | proteklo 15.4 s
Najbolji cilj: 11.579865143
Max|grad| te tačke: 1.743e-02
Iteracija 325 | proteklo 16.7 s
Najbolji cilj: 11.560906571
Max|grad| te tačke: 5.885e-03
Iteracija 350 | proteklo 18.0 s
Najbolji cilj: 11.548725889
Max|grad| te tačke: 5.101e-03
Iteracija 375 | proteklo 19.2 s
Najbolji cilj: 11.540184757
Max|grad| te tačke: 1.156e-02
Iteracija 400 | proteklo 20.5 s
Najbolji cilj: 11.536921027
Max|grad| te tačke: 4.457e-03
Iteracija 425 | proteklo 21.9 s
Najbolji cilj: 11.533269307
Max|grad| te tačke: 4.969e-03
Iteracija 450 | proteklo 23.1 s
Najbolji cilj: 11.528344480
Max|grad| te tačke: 1.571e-02
Iteracija 475 | proteklo 24.4 s
Najbolji cilj: 11.515786985
Max|grad| te tačke: 6.254e-03
Iteracija 500 | proteklo 25.7 s
Najbolji cilj: 11.510181076
Max|grad| te tačke: 9.906e-03
Iteracija 525 | proteklo 27.0 s
Najbolji cilj: 11.506636038
Max|grad| te tačke: 8.015e-03
Iteracija 550 | proteklo 28.3 s
Najbolji cilj: 11.502735301
Max|grad| te tačke: 7.497e-03
Iteracija 575 | proteklo 29.6 s
Najbolji cilj: 11.498425473
Max|grad| te tačke: 4.296e-03
Iteracija 600 | proteklo 31.0 s
Najbolji cilj: 11.497589594
Max|grad| te tačke: 1.131e-03
Zadržana validna tačka: 11.497589594
Max|grad|: 1.131e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 31.1 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.01 | početak=11 | prelaza=1212
Iteracija 25 | proteklo 1.5 s
Najbolji cilj: 13.436448047
Max|grad| te tačke: 2.262e-01
Iteracija 50 | proteklo 2.8 s
Najbolji cilj: 12.502458558
Max|grad| te tačke: 3.642e-02
Iteracija 75 | proteklo 4.1 s
Najbolji cilj: 12.117003257
Max|grad| te tačke: 6.616e-02
Iteracija 100 | proteklo 5.4 s
Najbolji cilj: 11.928108069
Max|grad| te tačke: 3.903e-02
Iteracija 125 | proteklo 6.6 s
Najbolji cilj: 11.811050555
Max|grad| te tačke: 1.864e-02
Iteracija 150 | proteklo 7.9 s
Najbolji cilj: 11.732687936
Max|grad| te tačke: 4.700e-02
Iteracija 175 | proteklo 9.2 s
Najbolji cilj: 11.662423857
Max|grad| te tačke: 2.512e-02
Iteracija 200 | proteklo 10.4 s
Najbolji cilj: 11.636380328
Max|grad| te tačke: 1.371e-02
Iteracija 225 | proteklo 11.7 s
Najbolji cilj: 11.614280677
Max|grad| te tačke: 2.693e-02
Iteracija 250 | proteklo 13.0 s
Najbolji cilj: 11.593321499
Max|grad| te tačke: 2.201e-02
Iteracija 275 | proteklo 14.3 s
Najbolji cilj: 11.583415536
Max|grad| te tačke: 1.822e-02
Iteracija 300 | proteklo 15.6 s
Najbolji cilj: 11.569881362
Max|grad| te tačke: 7.806e-03
Iteracija 325 | proteklo 16.8 s
Najbolji cilj: 11.555414952
Max|grad| te tačke: 1.392e-02
Iteracija 350 | proteklo 18.1 s
Najbolji cilj: 11.538643509
Max|grad| te tačke: 1.081e-02
Iteracija 375 | proteklo 19.4 s
Najbolji cilj: 11.521688993
Max|grad| te tačke: 7.887e-03
Iteracija 400 | proteklo 20.7 s
Najbolji cilj: 11.511181905
Max|grad| te tačke: 8.085e-03
Iteracija 425 | proteklo 22.0 s
Najbolji cilj: 11.504049377
Max|grad| te tačke: 4.788e-03
Iteracija 450 | proteklo 23.3 s
Najbolji cilj: 11.498223932
Max|grad| te tačke: 8.959e-03
Iteracija 475 | proteklo 24.5 s
Najbolji cilj: 11.488683415
Max|grad| te tačke: 8.424e-03
Iteracija 500 | proteklo 25.9 s
Najbolji cilj: 11.481968610
Max|grad| te tačke: 3.739e-03
Iteracija 525 | proteklo 27.2 s
Najbolji cilj: 11.475820212
Max|grad| te tačke: 1.271e-02
Iteracija 550 | proteklo 28.6 s
Najbolji cilj: 11.466304497
Max|grad| te tačke: 4.610e-03
Iteracija 575 | proteklo 29.8 s
Najbolji cilj: 11.459187275
Max|grad| te tačke: 4.193e-03
Iteracija 600 | proteklo 31.2 s
Najbolji cilj: 11.457886448
Max|grad| te tačke: 5.615e-03
Zadržana validna tačka: 11.457886448
Max|grad|: 5.615e-03
Gradijentni uslov: NIJE dostignut
OPTIMIZER_NOT_CONFIRMED: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Trajanje ovog poziva: 31.3 s
DIJAGNOSTIKA: konvergencija nije potvrđena.
Tačka nije proglašena lokalnim minimumom.
Validacioni postupak nastavlja sa njenom ispravnom raspodelom.
Validacioni NLL: 24.369843070
KANDIDAT: lag=3 | λ=0.01
Zbirni validacioni NLL: 24.133275030
Podela bez potvrđenog gradijentnog uslova: 1

IZABRANO: lag=1 | λ=0.01
Validacioni NLL: 18.918503889

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=1 | λ=0.01 | početak=17 | prelaza=1385
Iteracija 25 | proteklo 1.5 s
Najbolji cilj: 16.202855432
Max|grad| te tačke: 1.241e-02
Iteracija 50 | proteklo 2.9 s
Najbolji cilj: 16.175666047
Max|grad| te tačke: 7.510e-03
Iteracija 75 | proteklo 4.4 s
Najbolji cilj: 16.166103831
Max|grad| te tačke: 5.947e-03
Iteracija 100 | proteklo 5.8 s
Najbolji cilj: 16.163486297
Max|grad| te tačke: 9.421e-04
Iteracija 125 | proteklo 7.2 s
Najbolji cilj: 16.163365458
Max|grad| te tačke: 8.444e-04
Iteracija 150 | proteklo 8.8 s
Najbolji cilj: 16.162705961
Max|grad| te tačke: 1.642e-04
Iteracija 175 | proteklo 10.1 s
Najbolji cilj: 16.162702448
Max|grad| te tačke: 2.722e-05
Iteracija 200 | proteklo 11.6 s
Najbolji cilj: 16.162702371
Max|grad| te tačke: 3.412e-06
Iteracija 225 | proteklo 13.0 s
Najbolji cilj: 16.162702366
Max|grad| te tačke: 6.915e-07
Zadržana validna tačka: 16.162702366
Max|grad|: 6.915e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 13.1 s

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=1 | λ=0.01 | početak=11 | prelaza=1385
Iteracija 25 | proteklo 1.5 s
Najbolji cilj: 16.204043478
Max|grad| te tačke: 9.495e-03
Iteracija 50 | proteklo 2.9 s
Najbolji cilj: 16.175037598
Max|grad| te tačke: 3.989e-03
Iteracija 75 | proteklo 4.3 s
Najbolji cilj: 16.164814576
Max|grad| te tačke: 4.697e-03
Iteracija 100 | proteklo 5.7 s
Najbolji cilj: 16.163271377
Max|grad| te tačke: 1.137e-03
Iteracija 125 | proteklo 7.1 s
Najbolji cilj: 16.162705061
Max|grad| te tačke: 1.369e-04
Iteracija 150 | proteklo 8.5 s
Najbolji cilj: 16.162702471
Max|grad| te tačke: 4.658e-05
Iteracija 175 | proteklo 9.9 s
Najbolji cilj: 16.162702375
Max|grad| te tačke: 1.056e-05
Iteracija 200 | proteklo 11.3 s
Najbolji cilj: 16.162702366
Max|grad| te tačke: 1.771e-06
Zadržana validna tačka: 16.162702366
Max|grad|: 6.142e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 11.6 s
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 17.844460572

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=1 | λ=0.01 | početak=17 | prelaza=1732
Iteracija 25 | proteklo 1.9 s
Najbolji cilj: 16.354256998
Max|grad| te tačke: 9.647e-03
Iteracija 50 | proteklo 3.6 s
Najbolji cilj: 16.343712311
Max|grad| te tačke: 2.052e-03
Iteracija 75 | proteklo 5.4 s
Najbolji cilj: 16.340678161
Max|grad| te tačke: 2.164e-03
Iteracija 100 | proteklo 7.1 s
Najbolji cilj: 16.338982626
Max|grad| te tačke: 5.403e-04
Iteracija 125 | proteklo 8.9 s
Najbolji cilj: 16.338719559
Max|grad| te tačke: 5.573e-04
Iteracija 150 | proteklo 10.6 s
Najbolji cilj: 16.338693639
Max|grad| te tačke: 5.470e-05
Iteracija 175 | proteklo 12.3 s
Najbolji cilj: 16.338693294
Max|grad| te tačke: 1.087e-05
Iteracija 200 | proteklo 14.0 s
Najbolji cilj: 16.338693287
Max|grad| te tačke: 9.188e-07
Zadržana validna tačka: 16.338693287
Max|grad|: 9.188e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 14.2 s

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=1 | λ=0.01 | početak=11 | prelaza=1732
Iteracija 25 | proteklo 2.0 s
Najbolji cilj: 16.356133298
Max|grad| te tačke: 7.190e-03
Iteracija 50 | proteklo 3.7 s
Najbolji cilj: 16.344029129
Max|grad| te tačke: 2.002e-03
Iteracija 75 | proteklo 5.5 s
Najbolji cilj: 16.341707276
Max|grad| te tačke: 3.815e-03
Iteracija 100 | proteklo 7.3 s
Najbolji cilj: 16.339030819
Max|grad| te tačke: 1.074e-03
Iteracija 125 | proteklo 9.0 s
Najbolji cilj: 16.338870576
Max|grad| te tačke: 7.453e-04
Iteracija 150 | proteklo 10.8 s
Najbolji cilj: 16.338695006
Max|grad| te tačke: 1.306e-04
Iteracija 175 | proteklo 12.6 s
Najbolji cilj: 16.338693297
Max|grad| te tačke: 7.866e-06
Zadržana validna tačka: 16.338693287
Max|grad|: 8.758e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 14.4 s

loto7_4710_k86_loto_plus_1733.csv
NEXT: 08 x 25 y 31 z 39
Validacioni NLL: 18.918503889
Odvojeni test NLL: 17.844460572
Modelna verovatnoća NEXT: 6.35486773073e-06
Zbir uslovnih margina: 7.000000000000
Greška prema Qiskitu: 2.238e-16
Osetljivost uslovnih margina: 8.494235290e-01
Max|grad| završnog modela: 9.188e-07
Gradijentni uslov: dostignut



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 03 x 08 y 14 z 29
Validacioni NLL: 16.914203078
Odvojeni test NLL: 16.794391538
Modelna verovatnoća NEXT: 1.23357224444e-06
Zbir uslovnih margina: 7.000000000000
Greška prema Qiskitu: 2.238e-16
Osetljivost uslovnih margina: 2.166291096e-01
Max|grad| završnog modela: 6.060e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv
NEXT: 01 x 07 y 18 z 34
Validacioni NLL: 17.193813372
Odvojeni test NLL: 17.180689619
Modelna verovatnoća NEXT: 9.84990537534e-07
Zbir uslovnih margina: 7.000000000000
Greška prema Qiskitu: 2.242e-16
Osetljivost uslovnih margina: 2.581025450e-01
Max|grad| završnog modela: 6.513e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv
NEXT: 08 x 25 y 31 z 39
Validacioni NLL: 18.918503889
Odvojeni test NLL: 17.844460572
Modelna verovatnoća NEXT: 6.35486773073e-06
Zbir uslovnih margina: 7.000000000000
Greška prema Qiskitu: 2.238e-16
Osetljivost uslovnih margina: 8.494235290e-01
Max|grad| završnog modela: 9.188e-07
Gradijentni uslov: dostignut

Obrada je završena.
NEXT je tačan maksimum raspodele za sačuvane parametre.
Nepotvrđena konvergencija ostaje označena u rezultatu.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
V2 već ima drugačiju konstrukciju od V1: 
istorija upravlja Aliceinim rotacijama, a uči se i Bobov ortogonalni merni okvir. 
Radi vremena izvršavanja, trening će koristiti vektorizovanu propagaciju kompleksnih amplituda istih kola; 
ugrađene provere porediće je sa Qiskit Statevector, a završni NEXT će se izračunati i direktnim Qiskit izvršavanjem.

V2 zadržava istorijski upravljane Aliceine rotacije i naučeni Bobov merni okvir. 

Radi vremena izvršavanja, trening koristi vektorizovanu simulaciju kompleksnih amplituda istih Qiskit kola. 
Samoprovere porede taj proračun sa Statevector, a završni NEXT koristi direktno Qiskit izvršavanje. Nema klasičnog izlaznog regresora.
"""
