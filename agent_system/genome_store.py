"""Persistence for genomes, regulator states, and the cognitive runtime weights."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .cognitive_pipeline import CognitiveRuntime, RegulatorState
from .genome import PersonalityGenome

log = logging.getLogger(__name__)

GENOME_DIR_NAME    = 'genomes'
REGULATOR_DIR_NAME = 'regulator_states'
WEIGHTS_DIR_NAME   = 'cognitive_weights'


class GenomeStore:
    def __init__(self, memory_root: Path) -> None:
        self.genome_dir    = memory_root / GENOME_DIR_NAME
        self.regulator_dir = memory_root / REGULATOR_DIR_NAME
        self.weights_dir   = memory_root / WEIGHTS_DIR_NAME
        self.genome_dir.mkdir(parents=True, exist_ok=True)
        self.regulator_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    # ── Genome ────────────────────────────────────────────────────────────────

    def _genome_path(self, persona_id: str) -> Path:
        return self.genome_dir / f'{persona_id}.json'

    def load_genome(self, persona_id: str) -> PersonalityGenome:
        path = self._genome_path(persona_id)
        if path.exists():
            try:
                return PersonalityGenome.load(path)
            except Exception as exc:
                log.warning('genome load failed for %s: %s', persona_id, exc)
        return PersonalityGenome.default_for(persona_id)

    def save_genome(self, genome: PersonalityGenome) -> None:
        genome.save(self._genome_path(genome.persona_id))

    def list_genomes(self) -> list[str]:
        return [p.stem for p in self.genome_dir.glob('*.json')]

    # ── Regulator state ───────────────────────────────────────────────────────

    def _reg_path(self, session_id: str) -> Path:
        return self.regulator_dir / f'{session_id}.json'

    def load_regulator(self, session_id: str, genome: PersonalityGenome) -> RegulatorState:
        path = self._reg_path(session_id)
        if path.exists():
            try:
                return RegulatorState.load(path)
            except Exception as exc:
                log.warning('regulator load failed for %s: %s', session_id, exc)
        return RegulatorState.from_genome(genome)

    def save_regulator(self, session_id: str, state: RegulatorState) -> None:
        state.save(self._reg_path(session_id))

    # ── Runtime weights ───────────────────────────────────────────────────────

    def load_runtime(self) -> CognitiveRuntime:
        rt = CognitiveRuntime()
        rt.load(self.weights_dir)
        return rt

    def save_runtime(self, rt: CognitiveRuntime) -> None:
        rt.save(self.weights_dir)
