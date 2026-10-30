#!/usr/bin/env python3

# Quantum_pogadjanje


"""
QUANTUM POGADJANJE V3 — Qiskit EARAC codebook — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Preserved V3 construction:
    thirteen historical occupancy triples;
    eight codewords per group;
    two learned angular offsets per codeword;
    208 learned quantum parameters;
    fixed Bob X/Y/Z measurement axes;
    one classical Alice message;
    fresh singlet preparation for each query.

Qiskit computes the quantum codebook at every objective evaluation.
Historical samples select the appropriate quantum codebook entries.

No classical output regressor.
No PCA or learned classical mixture.

Independent fresh decoded outputs are conditioned on exactly
seven positive outputs.

The resulting seven-number distribution is normalized exactly.
Its exact maximum uses the seven largest Born log-odds.

The initial angular setting is an optimization starting point.
It is not an imposed law of future draws.

Optimizer limits are reported separately from numerical validity.
Best valid parameters and completed fits are checkpointed.

No random sampling, injected noise, row shuffling, rounding,
duplicate repair or cosmetic filters.
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
from zipfile import BadZipFile

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("//data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBER_COUNT = 39
DRAW_SIZE = 7
GROUP_COUNT = 13

RIDGES = (0.03, 0.3, 3.0)
INITIALIZATIONS = (0, 1)

OPTIMIZER_STEPS = 500
GRADIENT_LIMIT = 5e-5

CHECKPOINT_VERSION = "QISKIT_EARAC_CODEBOOK_V3_20261029"

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "Quantum_pogadjanje_v3_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class QuantumCodebook:
    """
    Qubit 0: Alice.
    Qubit 1: Bob.

    Corrected codeword direction:
        r = (sin(theta)*cos(phi),
             sin(theta)*sin(phi),
             cos(theta))

    Alice measurement direction:
        a = (-1)^first_history_bit * r

    Alice message:
        message = first_history_bit XOR Alice_Z_outcome

    Bob's decoded bit:
        guess = message XOR (1 - Bob_axis_outcome)

    For a singlet:
        P(guess=1) = (1 - r_axis)/2

    Qiskit amplitudes determine the probabilities.
    The last identity supplies the analytic circuit derivative
    and an independent consistency check.
    """

    def __init__(self, groups):
        self.groups = int(groups)
        self.m = 3 * self.groups
        self.parameter_count = self.groups * 8 * 2

        require(self.groups >= 1, "Neispravan broj grupa.")

        self.base_theta = math.acos(1.0 / math.sqrt(3.0))
        self.base_phi = math.pi / 4.0

        self.bits = np.array(
            [
                [(code >> 2) & 1, (code >> 1) & 1, code & 1]
                for code in range(8)
            ],
            dtype=np.int64,
        )

        self.first_sign = 1.0 - 2.0 * self.bits[:, 0]

        self.observables = (
            Pauli("XZ"),
            Pauli("YZ"),
            Pauli("ZZ"),
        )

        circuit = QuantumCircuit(2, name="EARACSinglet")
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
            "Neispravni parametri kvantne tabele.",
        )

        delta = parameters.reshape(self.groups, 8, 2)

        return (
            self.base_theta + delta[:, :, 0],
            self.base_phi + delta[:, :, 1],
        )

    def alice_state(self, theta, phi, code):
        # Adding pi reverses the whole measurement direction.
        physical_theta = (
            float(theta)
            + math.pi * int(self.bits[code, 0])
        )

        circuit = QuantumCircuit(2, name="AliceCodewordBasis")
        circuit.rz(-float(phi), 0)
        circuit.ry(-physical_theta, 0)

        return self.singlet.evolve(circuit)

    def lookup(self, parameters, with_gradient=False):
        theta, phi = self.angles(parameters)

        q = np.empty((self.groups, 8, 3), dtype=float)
        r = np.empty_like(q)

        for group in range(self.groups):
            for code in range(8):
                state = self.alice_state(
                    theta[group, code],
                    phi[group, code],
                    code,
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
                        "Singlet korelacija ima imaginarni ostatak.",
                    )

                    decoded_bias = (
                        self.first_sign[code] * correlation.real
                    )

                    q[group, code, axis] = (
                        1.0 + decoded_bias
                    ) / 2.0

                    r[group, code, axis] = (
                        1.0 - decoded_bias
                    ) / 2.0

        require(
            np.all(np.isfinite(q))
            and np.all(np.isfinite(r))
            and np.all(q > 0)
            and np.all(r > 0),
            "Nulta ili nevažeća Born verovatnoća u kvantnoj tabeli.",
        )

        require(
            np.max(np.abs(q + r - 1.0)) < 2e-10,
            "Kvantna tabela nije normalizovana.",
        )

        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)

        directions = np.stack(
            (st * cp, st * sp, ct),
            axis=-1,
        )

        reference_q = (1.0 - directions) / 2.0

        require(
            np.max(np.abs(q - reference_q)) < 2e-10,
            "Qiskit Born izlaz nije saglasan sa EARAC konstrukcijom.",
        )

        require(
            np.max(np.abs(
                np.sum((r - q) ** 2, axis=2) - 1.0
            )) < 2e-9,
            "Narušena je sferna kvantna granica.",
        )

        if not with_gradient:
            return q, r, None

        derivative_theta = np.stack(
            (ct * cp, ct * sp, -st),
            axis=-1,
        )

        derivative_phi = np.stack(
            (-st * sp, st * cp, np.zeros_like(st)),
            axis=-1,
        )

        derivatives = -0.5 * np.stack(
            (derivative_theta, derivative_phi),
            axis=-1,
        )

        return q, r, derivatives

    def verify_explicit_messages(self, parameters):
        theta, phi = self.angles(parameters)
        q, _, _ = self.lookup(parameters)

        largest_error = 0.0

        for group in range(self.groups):
            for code in range(8):
                alice_state = self.alice_state(
                    theta[group, code],
                    phi[group, code],
                    code,
                )

                for axis in range(3):
                    readout = QuantumCircuit(2)

                    if axis == 0:
                        readout.h(1)
                    elif axis == 1:
                        readout.sdg(1)
                        readout.h(1)

                    state = alice_state.evolve(readout)
                    probabilities = np.abs(state.data) ** 2

                    require(
                        abs(float(probabilities.sum()) - 1.0) < 1e-11,
                        "Eksplicitne Born verovatnoće nisu normalizovane.",
                    )

                    probability_one = 0.0

                    for index, probability in enumerate(probabilities):
                        alice_outcome = index & 1
                        bob_outcome = (index >> 1) & 1

                        message = (
                            int(self.bits[code, 0])
                            ^ alice_outcome
                        )

                        guess = message ^ (1 - bob_outcome)

                        if guess == 1:
                            probability_one += float(probability)

                    largest_error = max(
                        largest_error,
                        abs(probability_one - q[group, code, axis]),
                    )

        require(
            largest_error < 2e-10,
            "Eksplicitna Aliceina poruka nije saglasna sa tabelom.",
        )

        return largest_error


def codes_from_occupancy(occupancy):
    occupancy = np.asarray(occupancy)

    require(
        occupancy.ndim == 2
        and occupancy.shape[1] % 3 == 0
        and np.all((occupancy == 0) | (occupancy == 1)),
        "Neispravne istorijske binarne oznake.",
    )

    triples = occupancy.astype(np.int64).reshape(
        len(occupancy),
        occupancy.shape[1] // 3,
        3,
    )

    return (
        4 * triples[:, :, 0]
        + 2 * triples[:, :, 1]
        + triples[:, :, 2]
    )


def select_probabilities(q_lookup, r_lookup, codes):
    groups = q_lookup.shape[0]

    require(
        codes.ndim == 2
        and codes.shape[1] == groups
        and np.all((codes >= 0) & (codes < 8)),
        "Neispravni indeksi kvantne tabele.",
    )

    group_indices = np.arange(groups)[None, :]

    q = q_lookup[group_indices, codes].reshape(
        len(codes), groups * 3
    )

    r = r_lookup[group_indices, codes].reshape(
        len(codes), groups * 3
    )

    return q, r


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
            fields[:, number, None] + forward[:, number, :-1],
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
        "Uslovne marginalne verovatnoće nisu ispravne.",
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
    source,
    codes,
    targets,
    ridge,
    gradient=True,
):
    q_lookup, r_lookup, derivatives = source.lookup(
        parameters,
        with_gradient=gradient,
    )

    q, r = select_probabilities(
        q_lookup,
        r_lookup,
        codes,
    )

    fields = np.log(q) - np.log(r)
    k = targets.shape[1]

    log_z, marginals = fixed_size_partition(
        fields,
        k,
        with_marginals=gradient,
    )

    rows = np.arange(len(targets))[:, None]
    scores = fields[rows, targets].sum(axis=1)
    logp = scores - log_z

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća posmatrane kombinacije.",
    )

    loss = (
        -float(logp.mean())
        + ridge * float(np.dot(parameters, parameters))
    )

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    observed = np.zeros_like(q)
    observed[rows, targets] = 1.0

    probability_gradient = (
        (marginals - observed) / (q * r) / len(targets)
    ).reshape(len(targets), source.groups, 3)

    lookup_gradient = np.zeros_like(q_lookup)

    for group in range(source.groups):
        np.add.at(
            lookup_gradient[group],
            codes[:, group],
            probability_gradient[:, group, :],
        )

    angular_gradient = np.sum(
        lookup_gradient[:, :, :, None] * derivatives,
        axis=2,
    )

    result = (
        angular_gradient.ravel()
        + 2.0 * ridge * parameters
    )

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


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

    require(
        len(rows) >= 100,
        f"{path.name}: nedovoljno istorijskih kola.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.int64,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def checkpoint_path(codes, targets, ridge, initialization):
    digest = hashlib.sha256()

    configuration = (
        CHECKPOINT_VERSION,
        float(ridge),
        int(initialization),
        OPTIMIZER_STEPS,
        qiskit.__version__,
        np.__version__,
    )

    digest.update(repr(configuration).encode("utf-8"))

    for array in (codes, targets):
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

    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        EOFError,
        BadZipFile,
    ) as error:
        say(
            f"Kontrolna tačka nije učitana: {path.name}",
            str(error),
        )

        return None


@dataclass
class Model:
    source: QuantumCodebook
    parameters: np.ndarray
    q_lookup: np.ndarray
    r_lookup: np.ndarray
    ridge: float
    loss: float
    gradient_norm: float
    converged: bool
    optimizer_message: str

    def probabilities(self, codes):
        return select_probabilities(
            self.q_lookup,
            self.r_lookup,
            codes,
        )


def fit(codes, targets, ridge, label):
    source = QuantumCodebook(codes.shape[1])
    accepted = []

    for initialization in INITIALIZATIONS:
        if initialization == 0:
            initial = np.zeros(source.parameter_count)
        else:
            initial = 0.005 * np.sin(
                np.arange(source.parameter_count) + 0.7
            )

        path = checkpoint_path(
            codes,
            targets,
            ridge,
            initialization,
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
            f"λ={ridge:g} | početak={initialization}"
            f" | prelaza={len(targets)}",
            f"Kvantnih ugaonih parametara: {source.parameter_count}",
        )

        try:
            best_loss, best_gradient = objective(
                parameters,
                source,
                codes,
                targets,
                ridge,
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            if saved is None:
                say(
                    f"Početak {initialization} nije numerički validan:",
                    str(error),
                )
                continue

            say(
                "Sačuvana tačka nije prošla proveru.",
                "Koristim deterministički početak.",
            )

            parameters = initial.copy()
            used_iterations = 0
            already_done = False

            best_loss, best_gradient = objective(
                parameters,
                source,
                codes,
                targets,
                ridge,
            )

        best_parameters = parameters.copy()
        best_gradient = best_gradient.copy()

        added_iterations = 0
        optimizer_success = False

        if already_done:
            say("Učitan završen proračun iz kontrolne tačke.")

        else:
            remaining = max(
                0,
                OPTIMIZER_STEPS - used_iterations,
            )

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
                    current,
                    source,
                    codes,
                    targets,
                    ridge,
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
                    total_iterations = (
                        used_iterations + added_iterations
                    )

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

        checked_loss, checked_gradient = objective(
            best_parameters,
            source,
            codes,
            targets,
            ridge,
        )

        maximum_gradient = float(
            np.max(np.abs(checked_gradient))
        )

        q_lookup, r_lookup, _ = source.lookup(best_parameters)

        converged = maximum_gradient <= GRADIENT_LIMIT

        if not already_done:
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

        accepted.append(
            Model(
                source=source,
                parameters=best_parameters.copy(),
                q_lookup=q_lookup.copy(),
                r_lookup=r_lookup.copy(),
                ridge=ridge,
                loss=float(checked_loss),
                gradient_norm=maximum_gradient,
                converged=converged,
                optimizer_message=message,
            )
        )

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
                "Validacioni postupak nastavlja sa njenom validnom raspodelom.",
            )

    require(
        bool(accepted),
        f"{label}: nema nijedne numerički ispravne tačke.",
    )

    return min(accepted, key=lambda model: model.loss)


def exact_next(fields, k):
    fields = np.asarray(fields, dtype=float)

    require(
        fields.ndim == 1
        and np.all(np.isfinite(fields))
        and 1 <= k <= len(fields),
        "Neispravna polja za izbor maksimuma.",
    )

    order = sorted(
        range(len(fields)),
        key=lambda number: (-float(fields[number]), number),
    )

    return tuple(sorted(order[:k]))


def evaluate(model, codes, targets, hits=False):
    q, r = model.probabilities(codes)
    fields = np.log(q) - np.log(r)

    log_z, _ = fixed_size_partition(
        fields,
        targets.shape[1],
    )

    rows = np.arange(len(targets))[:, None]
    logp = fields[rows, targets].sum(axis=1) - log_z

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna evaluaciona verovatnoća.",
    )

    histogram = np.zeros(
        targets.shape[1] + 1,
        dtype=np.int64,
    )

    if hits:
        for sample in range(len(targets)):
            prediction = exact_next(
                fields[sample],
                targets.shape[1],
            )

            matched = len(
                set(prediction).intersection(
                    targets[sample].tolist()
                )
            )

            histogram[matched] += 1

        require(
            int(histogram.sum()) == len(targets),
            "Histogram pogodaka nije potpun.",
        )

    return -float(logp.mean()), histogram


def self_checks():
    say(
        "",
        "Pokrećem ugrađene Qiskit i matematičke provere.",
    )

    started = time.perf_counter()

    source = QuantumCodebook(2)

    parameters = 0.04 * np.sin(
        np.arange(source.parameter_count) + 0.7
    )

    message_error = source.verify_explicit_messages(parameters)

    # Check the original optimal three-bit EARAC codewords separately.
    reference = QuantumCodebook(1)
    reference_delta = np.zeros((1, 8, 2))

    for code in range(8):
        signs = 1.0 - 2.0 * reference.bits[code]
        direction = signs / math.sqrt(3.0)

        desired_theta = math.acos(float(direction[2]))
        desired_phi = math.atan2(
            float(direction[1]),
            float(direction[0]),
        )

        reference_delta[0, code, 0] = (
            desired_theta - reference.base_theta
        )

        reference_delta[0, code, 1] = (
            desired_phi - reference.base_phi
        )

    reference_q, reference_r, _ = reference.lookup(
        reference_delta.ravel()
    )

    success = np.where(
        reference.bits == 1,
        reference_q[0],
        reference_r[0],
    )

    require(
        np.max(np.abs(
            success
            - (1.0 + 1.0 / math.sqrt(3.0)) / 2.0
        )) < 1e-11,
        "Originalni trobitni EARAC protokol nije prošao.",
    )

    initial_q, _, _ = source.lookup(
        np.zeros(source.parameter_count)
    )

    require(
        np.max(np.abs(
            initial_q - (1.0 - 1.0 / math.sqrt(3.0)) / 2.0
        )) < 1e-11,
        "Početno ugaono podešavanje nije reprodukovano.",
    )

    codes = np.array(
        [[0, 1], [2, 4], [5, 3], [7, 6]],
        dtype=np.int64,
    )

    targets = np.array(
        [[0, 4], [1, 3], [2, 5], [0, 5]],
        dtype=np.int64,
    )

    ridge = 0.03

    _, analytic = objective(
        parameters,
        source,
        codes,
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
                positive,
                source,
                codes,
                targets,
                ridge,
                gradient=False,
            )
            - objective(
                negative,
                source,
                codes,
                targets,
                ridge,
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

    q_lookup, r_lookup, _ = source.lookup(parameters)
    q, r = select_probabilities(q_lookup, r_lookup, codes)

    fields = np.log(q) - np.log(r)

    log_z, marginals = fixed_size_partition(
        fields,
        2,
        with_marginals=True,
    )

    subsets = list(combinations(range(6), 2))
    indices = np.asarray(subsets, dtype=np.int64)

    for sample in range(len(codes)):
        scores = fields[sample][indices].sum(axis=1)
        direct_log_z = float(logsumexp(scores))
        probabilities = np.exp(scores - direct_log_z)

        require(
            abs(direct_log_z - log_z[sample]) < 1e-11
            and abs(float(probabilities.sum()) - 1.0) < 1e-12,
            "Potpuna provera normalizacije nije prošla.",
        )

        direct_marginals = np.zeros(6)

        for selected, probability in zip(subsets, probabilities):
            direct_marginals[np.asarray(selected)] += probability

        require(
            np.max(np.abs(
                direct_marginals - marginals[sample]
            )) < 1e-11,
            "Potpuna provera margina nije prošla.",
        )

        require(
            exact_next(fields[sample], 2)
            == subsets[int(np.argmax(scores))],
            "Provera tačnog maksimuma nije prošla.",
        )

    for k in (1, 6):
        z, marginal = fixed_size_partition(
            np.zeros((1, 6)),
            k,
            with_marginals=True,
        )

        require(
            abs(z[0] - math.log(math.comb(6, k))) < 1e-11
            and np.max(np.abs(marginal[0] - k / 6)) < 1e-11,
            "Granična provera nije prošla.",
        )

    require(
        exact_next(np.zeros(6), 2) == (0, 1),
        "Obrada jednakih maksimuma nije prošla.",
    )

    toy_occupancy = np.array(
        [[0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0]],
        dtype=np.int64,
    )

    require(
        np.array_equal(
            codes_from_occupancy(toy_occupancy),
            np.array([[0, 1], [4, 6]]),
        ),
        "Trobitno kodiranje nije prošlo.",
    )

    say(
        "Ugrađene provere su prošle.",
        f"Greška eksplicitne Aliceine poruke: {message_error:.3e}",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
        "Provereni su protokol, normalizacija, margine i maksimum.",
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
        f"Test kola: {result['test_count']}",
        "Pogodaka 0–7: "
        + " ".join(str(int(value)) for value in result["histogram"]),
        f"Prosečan broj pogodaka na testu: {result['mean_hits']:.6f}",
        f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
        f"Zbir uslovnih margina: {result['marginal_sum']:.12f}",
        f"Greška eksplicitne poruke: {result['message_error']:.3e}",
        f"Osetljivost uslovnih margina: {result['sensitivity']:.9e}",
        f"Max|grad| završnog modela: {result['gradient']:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if result["converged"] else "NIJE dostignut"),
    )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process(path, draws, occupancy, fingerprint):
    all_codes = codes_from_occupancy(occupancy)

    # Each input precedes its target by exactly one draw.
    codes = all_codes[:-1]
    targets = draws[1:]

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
            model = fit(
                codes[:train_end],
                targets[:train_end],
                ridge,
                f"{path.name} / validacija {fold}",
            )

            incomplete_folds += int(not model.converged)

            score = evaluate(
                model,
                codes[train_end:validation_end],
                targets[train_end:validation_end],
            )[0]

            count = validation_end - train_end
            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count

        scores.append(
            (score, ridge, incomplete_folds)
        )

        say(
            f"KANDIDAT: λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
            f"Podela bez potvrđenog gradijentnog uslova: {incomplete_folds}",
        )

    validation_nll, ridge, incomplete_folds = min(
        scores,
        key=lambda item: item[0],
    )

    say(
        "",
        f"IZABRANO: λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit(
        codes[:split80],
        targets[:split80],
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_model,
        codes[split80:],
        targets[split80:],
        hits=True,
    )

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"NLL cele sedmorke: {test_nll:.9f}",
        "Pogodaka 0–7: "
        + " ".join(str(int(value)) for value in histogram),
    )

    final_model = fit(
        codes,
        targets,
        ridge,
        f"{path.name} / završno učenje svih {n} prelaza",
    )

    # Re-execute the learned quantum codebook for final inference.
    q_lookup, r_lookup, _ = final_model.source.lookup(
        final_model.parameters
    )

    final_model.q_lookup = q_lookup
    final_model.r_lookup = r_lookup

    message_error = final_model.source.verify_explicit_messages(
        final_model.parameters
    )

    q, r = final_model.probabilities(all_codes[-1:])
    fields = np.log(q) - np.log(r)

    log_z, marginal = fixed_size_partition(
        fields,
        DRAW_SIZE,
        with_marginals=True,
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

    old_q, old_r = final_model.probabilities(codes[-8:])

    _, old_marginal = fixed_size_partition(
        np.log(old_q) - np.log(old_r),
        DRAW_SIZE,
        with_marginals=True,
    )

    sensitivity = float(
        np.max(np.abs(old_marginal - marginal[0]))
    )

    repeats = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
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

    if repeats:
        warnings.append(
            f"NEXT se pojavljuje {repeats} puta u istoriji."
        )

    if best == tuple(int(value) for value in draws[-1]):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Uslovne margine slabo reaguju na proverene kontekste."
        )

    result = {
        "name": path.name,
        "hash": fingerprint,
        "next": tuple(value + 1 for value in best),
        "validation": validation_nll,
        "test": test_nll,
        "histogram": histogram,
        "test_count": n - split80,
        "mean_hits": float(
            histogram @ np.arange(DRAW_SIZE + 1)
            / (n - split80)
        ),
        "probability": math.exp(logp),
        "marginal_sum": float(marginal.sum()),
        "message_error": message_error,
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

    require(
        NUMBER_COUNT == GROUP_COUNT * 3,
        "Broj oznaka nije usaglašen sa trobitnim grupama.",
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    say(
        "QUANTUM POGADJANJE V3 — Qiskit EARAC tabela — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Trinaest trobitnih grupa i 208 ugaonih parametara.",
        "Alice uči tabelu kvantnih kodnih pravaca.",
        "Bob koristi fiksna X/Y/Z merenja.",
        "Qiskit računa kvantnu tabelu pri svakoj oceni funkcije cilja.",
        "Izlaz nastaje iz singlet amplituda i Bornovih verovatnoća.",
        "Svaki upit predstavlja zasebnu pripremu.",
        "Nema klasičnog izlaznog regresora.",
        "Klasični optimizator podešava kvantne uglove.",
        "Ulaz jednog uzorka je prethodno kolo.",
        "Završno učenje koristi sve istorijske prelaze.",
        "Raspodela se tačno uslovljava na sedam različitih brojeva.",
        "Nema slučajnog uzorkovanja, šuma ili mešanja redova.",
        "Limit iteracija ne odbacuje validnu sačuvanu raspodelu.",
        "Konvergencija se prijavljuje odvojeno.",
        "Uslovljena raspodela nema dodatne naučene interakcije među izlaznim brojevima.",
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

        datasets.append(
            (path, draws, occupancy, fingerprint)
        )

        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, occupancy, fingerprint in datasets:
        results.append(
            process(path, draws, occupancy, fingerprint)
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
QUANTUM POGADJANJE V3 — Qiskit EARAC tabela — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Trinaest trobitnih grupa i 208 ugaonih parametara.
Alice uči tabelu kvantnih kodnih pravaca.
Bob koristi fiksna X/Y/Z merenja.
Qiskit računa kvantnu tabelu pri svakoj oceni funkcije cilja.
Izlaz nastaje iz singlet amplituda i Bornovih verovatnoća.
Svaki upit predstavlja zasebnu pripremu.
Nema klasičnog izlaznog regresora.
Klasični optimizator podešava kvantne uglove.
Ulaz jednog uzorka je prethodno kolo.
Završno učenje koristi sve istorijske prelaze.
Raspodela se tačno uslovljava na sedam različitih brojeva.
Nema slučajnog uzorkovanja, šuma ili mešanja redova.
Limit iteracija ne odbacuje validnu sačuvanu raspodelu.
Konvergencija se prijavljuje odvojeno.
Uslovljena raspodela nema dodatne naučene interakcije među izlaznim brojevima.
Budžet: najviše 500 iteracija po početku.
Kontrolne tačke: /Quantum_pogadjanje/Quantum_pogadjanje_v3_checkpoints

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
Greška eksplicitne Aliceine poruke: 2.498e-16
Najveća greška gradijenta: 3.722e-10
Provereni su protokol, normalizacija, margine i maksimum.
Trajanje provera: 0.2 s

loto7_4710_k86.csv / validacija 1
λ=0.03 | početak=0 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.532224003
Max|grad|: 5.120e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.4 s

loto7_4710_k86.csv / validacija 1
λ=0.03 | početak=1 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.532224003
Max|grad|: 7.490e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.4 s
Validacioni NLL: 16.557954606

loto7_4710_k86.csv / validacija 2
λ=0.03 | početak=0 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.534487002
Max|grad|: 8.069e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.4 s

loto7_4710_k86.csv / validacija 2
λ=0.03 | početak=1 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.534487002
Max|grad|: 3.859e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.6 s
Validacioni NLL: 16.555311384
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.556632995
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
λ=0.3 | početak=0 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.543805377
Max|grad|: 5.314e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.7 s

loto7_4710_k86.csv / validacija 1
λ=0.3 | početak=1 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.543805377
Max|grad|: 6.693e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.7 s
Validacioni NLL: 16.549488102

loto7_4710_k86.csv / validacija 2
λ=0.3 | početak=0 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.544439194
Max|grad|: 4.392e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.8 s

loto7_4710_k86.csv / validacija 2
λ=0.3 | početak=1 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.544439194
Max|grad|: 5.259e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.8 s
Validacioni NLL: 16.549029205
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.549258654
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86.csv / validacija 1
λ=3 | početak=0 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547941774
Max|grad|: 8.718e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s

loto7_4710_k86.csv / validacija 1
λ=3 | početak=1 | prelaza=2825
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547941774
Max|grad|: 1.567e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s
Validacioni NLL: 16.548558671

loto7_4710_k86.csv / validacija 2
λ=3 | početak=0 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548029943
Max|grad|: 6.268e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s

loto7_4710_k86.csv / validacija 2
λ=3 | početak=1 | prelaza=3296
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548029943
Max|grad|: 1.456e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s
Validacioni NLL: 16.548523031
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.548540851
Podela bez potvrđenog gradijentnog uslova: 0

IZABRANO: λ=3
Validacioni NLL: 16.548540851

loto7_4710_k86.csv / odvojeni test
λ=3 | početak=0 | prelaza=3767
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548057765
Max|grad|: 5.647e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s

loto7_4710_k86.csv / odvojeni test
λ=3 | početak=1 | prelaza=3767
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548057765
Max|grad|: 1.424e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s
ODVOJENI TEST: 942 prelaza
NLL cele sedmorke: 16.548333541
Pogodaka 0–7: 197 402 261 70 12 0 0 0

loto7_4710_k86.csv / završno učenje svih 4709 prelaza
λ=3 | početak=0 | prelaza=4709
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548107292
Max|grad|: 6.271e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.7 s

loto7_4710_k86.csv / završno učenje svih 4709 prelaza
λ=3 | početak=1 | prelaza=4709
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.548107292
Max|grad|: 1.378e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.8 s

loto7_4710_k86.csv
NEXT: 03 x 10 y 21 z 39
Validacioni NLL: 16.548540851
Odvojeni test NLL: 16.548333541
Test kola: 942
Pogodaka 0–7: 197 402 261 70 12 0 0 0
Prosečan broj pogodaka na testu: 1.254777
Modelna verovatnoća NEXT: 6.76454811946e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 3.608e-16
Osetljivost uslovnih margina: 2.194203891e-03
Max|grad| završnog modela: 1.378e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.03 | početak=0 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.517084394
Max|grad|: 6.711e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.1 s

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.03 | početak=1 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.517084394
Max|grad|: 8.970e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.0 s
Validacioni NLL: 16.567922437

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.03 | početak=0 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.521294200
Max|grad|: 4.650e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.2 s

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.03 | početak=1 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.521294200
Max|grad|: 4.941e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.2 s
Validacioni NLL: 16.567268945
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.567596241
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.3 | početak=0 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.539017996
Max|grad|: 1.749e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.3 | početak=1 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.539017996
Max|grad|: 1.671e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s
Validacioni NLL: 16.549624600

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.3 | početak=0 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.540145685
Max|grad|: 5.324e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.3 | početak=1 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.540145685
Max|grad|: 4.821e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.6 s
Validacioni NLL: 16.551027029
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.550324636
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_2977.csv / validacija 1
λ=3 | početak=0 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547242317
Max|grad|: 4.412e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_2977.csv / validacija 1
λ=3 | početak=1 | prelaza=1785
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547242317
Max|grad|: 1.570e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s
Validacioni NLL: 16.548345698

loto7_4710_k86_loto_2977.csv / validacija 2
λ=3 | početak=0 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547401911
Max|grad|: 3.612e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_2977.csv / validacija 2
λ=3 | početak=1 | prelaza=2083
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547401911
Max|grad|: 1.562e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s
Validacioni NLL: 16.548639032
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.548492119
Podela bez potvrđenog gradijentnog uslova: 0

IZABRANO: λ=3
Validacioni NLL: 16.548492119

loto7_4710_k86_loto_2977.csv / odvojeni test
λ=3 | početak=0 | prelaza=2380
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547574663
Max|grad|: 8.575e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_2977.csv / odvojeni test
λ=3 | početak=1 | prelaza=2380
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547574663
Max|grad|: 1.560e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s
ODVOJENI TEST: 596 prelaza
NLL cele sedmorke: 16.547968099
Pogodaka 0–7: 121 229 167 67 12 0 0 0

loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza
λ=3 | početak=0 | prelaza=2976
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547682807
Max|grad|: 6.016e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s

loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza
λ=3 | početak=1 | prelaza=2976
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547682807
Max|grad|: 1.539e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s

loto7_4710_k86_loto_2977.csv
NEXT: 10 x 16 y 22 z 39
Validacioni NLL: 16.548492119
Odvojeni test NLL: 16.547968099
Test kola: 596
Pogodaka 0–7: 121 229 167 67 12 0 0 0
Prosečan broj pogodaka na testu: 1.362416
Modelna verovatnoća NEXT: 6.83260474807e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 3.331e-16
Osetljivost uslovnih margina: 2.245956569e-03
Max|grad| završnog modela: 1.539e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.03 | početak=0 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.492958210
Max|grad|: 4.002e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.0 s

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.03 | početak=1 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.492958210
Max|grad|: 5.916e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.8 s
Validacioni NLL: 16.593957655

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.03 | početak=0 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.503553719
Max|grad|: 9.618e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.9 s

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.03 | početak=1 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.503553719
Max|grad|: 3.296e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 1.0 s
Validacioni NLL: 16.565949197
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.579953426
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.3 | početak=0 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.530724882
Max|grad|: 4.379e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.3 | početak=1 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.530724882
Max|grad|: 4.937e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s
Validacioni NLL: 16.556348749

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.3 | početak=0 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.534233441
Max|grad|: 9.670e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.3 | početak=1 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.534233441
Max|grad|: 9.676e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.5 s
Validacioni NLL: 16.549353670
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.552851210
Podela bez potvrđenog gradijentnog uslova: 0

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=3 | početak=0 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.545955053
Max|grad|: 1.023e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=3 | početak=1 | prelaza=1039
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.545955053
Max|grad|: 1.671e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
Validacioni NLL: 16.548990991

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=3 | početak=0 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.546484086
Max|grad|: 1.111e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=3 | početak=1 | prelaza=1212
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.546484086
Max|grad|: 1.602e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
Validacioni NLL: 16.548038394
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.548514692
Podela bez potvrđenog gradijentnog uslova: 0

IZABRANO: λ=3
Validacioni NLL: 16.548514692

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
λ=3 | početak=0 | prelaza=1385
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.546708397
Max|grad|: 1.290e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
λ=3 | početak=1 | prelaza=1385
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.546708397
Max|grad|: 1.620e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.3 s
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 16.549609316
Pogodaka 0–7: 76 152 90 22 6 1 0 0

loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza
λ=3 | početak=0 | prelaza=1732
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547252897
Max|grad|: 5.591e-08
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza
λ=3 | početak=1 | prelaza=1732
Kvantnih ugaonih parametara: 208
Zadržana validna tačka: 16.547252897
Max|grad|: 1.590e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Trajanje ovog poziva: 0.4 s

loto7_4710_k86_loto_plus_1733.csv
NEXT: 04 x 11 y 23 z 29
Validacioni NLL: 16.548514692
Odvojeni test NLL: 16.549609316
Test kola: 347
Pogodaka 0–7: 76 152 90 22 6 1 0 0
Prosečan broj pogodaka na testu: 1.230548
Modelna verovatnoća NEXT: 6.85983288084e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 4.163e-16
Osetljivost uslovnih margina: 2.394760327e-03
Max|grad| završnog modela: 5.591e-08
Gradijentni uslov: dostignut



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 03 x 10 y 21 z 39
Validacioni NLL: 16.548540851
Odvojeni test NLL: 16.548333541
Test kola: 942
Pogodaka 0–7: 197 402 261 70 12 0 0 0
Prosečan broj pogodaka na testu: 1.254777
Modelna verovatnoća NEXT: 6.76454811946e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 3.608e-16
Osetljivost uslovnih margina: 2.194203891e-03
Max|grad| završnog modela: 1.378e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_2977.csv
NEXT: 10 x 16 y 22 z 39
Validacioni NLL: 16.548492119
Odvojeni test NLL: 16.547968099
Test kola: 596
Pogodaka 0–7: 121 229 167 67 12 0 0 0
Prosečan broj pogodaka na testu: 1.362416
Modelna verovatnoća NEXT: 6.83260474807e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 3.331e-16
Osetljivost uslovnih margina: 2.245956569e-03
Max|grad| završnog modela: 1.539e-07
Gradijentni uslov: dostignut

loto7_4710_k86_loto_plus_1733.csv
NEXT: 04 x 11 y 23 z 29
Validacioni NLL: 16.548514692
Odvojeni test NLL: 16.549609316
Test kola: 347
Pogodaka 0–7: 76 152 90 22 6 1 0 0
Prosečan broj pogodaka na testu: 1.230548
Modelna verovatnoća NEXT: 6.85983288084e-08
Zbir uslovnih margina: 7.000000000000
Greška eksplicitne poruke: 4.163e-16
Osetljivost uslovnih margina: 2.394760327e-03
Max|grad| završnog modela: 5.591e-08
Gradijentni uslov: dostignut

Obrada je završena.
NEXT je tačan maksimum raspodele za sačuvane parametre.
Nepotvrđena konvergencija ostaje označena.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
V3 uči tabelu od 208 ugaonih parametara za 13 trobitnih grupa, dok su Bobova merenja fiksna. 
Qiskit će računati tabelu amplitudnih očitavanja pri svakoj oceni funkcije cilja; 
istorijski uzorci zatim koriste odgovarajuće unose, pa nije potrebno ponavljati ista kola za svako kolo.

V3 zadržava naučenu EARAC tabelu od 208 ugaonih parametara, prethodno kolo kao ulaz i fiksna Bobova merenja.

Ovde Qiskit računa Bornova očitavanja i tokom treninga, jednom po unosu tabele pri svakoj oceni funkcije cilja. 
Uključeno je čuvanje najboljih validnih parametara i nastavak preko kontrolnih tačaka. 
Limit iteracija prijavljuje se odvojeno od numeričke ispravnosti.
"""
