from __future__ import annotations

from collections.abc import Iterable

from gvi.models import EvidenceRef, Observation, Scene, TimeRange, Track, Zone, ZoneVisit


class SceneMemory:
    """Read-only deterministic queries over a validated visual evidence graph."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._tracks = {track.track_id: track for track in scene.tracks}
        self._zones = {zone.zone_id: zone for zone in scene.zones}

    def find_tracks(self, concept: str, time_range: TimeRange | None = None) -> tuple[Track, ...]:
        query = concept.casefold().strip()
        matches = (
            track
            for track in self.scene.tracks
            if query in track.concept.casefold()
            and (time_range is None or self._has_observation_in(track, time_range))
        )
        return tuple(matches)

    def count_unique(self, concept: str, time_range: TimeRange | None = None) -> int:
        return len(self.find_tracks(concept, time_range))

    def first_seen(self, track_id: str) -> EvidenceRef:
        track = self._track(track_id)
        observation = next(observation for observation in track.observations if observation.visible)
        return self._observation_evidence(track, observation, "first visible observation")

    def last_seen(self, track_id: str) -> EvidenceRef:
        track = self._track(track_id)
        observation = next(
            observation for observation in reversed(track.observations) if observation.visible
        )
        return self._observation_evidence(track, observation, "last visible observation")

    def zone_visits(self, track_id: str, zone_id: str) -> tuple[ZoneVisit, ...]:
        track = self._track(track_id)
        zone = self._zone(zone_id)
        observations = [
            observation
            for observation in track.observations
            if observation.visible and self._is_inside(observation, zone)
        ]
        groups = self._contiguous_groups(observations)
        return tuple(self._to_zone_visit(track, zone, group) for group in groups)

    def observed_dwell_time(self, track_id: str, zone_id: str) -> float:
        return sum(visit.observed_duration_s for visit in self.zone_visits(track_id, zone_id))

    def _track(self, track_id: str) -> Track:
        try:
            return self._tracks[track_id]
        except KeyError as error:
            raise KeyError(f"unknown track: {track_id}") from error

    def _zone(self, zone_id: str) -> Zone:
        try:
            return self._zones[zone_id]
        except KeyError as error:
            raise KeyError(f"unknown zone: {zone_id}") from error

    @staticmethod
    def _has_observation_in(track: Track, time_range: TimeRange) -> bool:
        return any(
            observation.visible and time_range.contains(observation.timestamp_s)
            for observation in track.observations
        )

    def _is_inside(self, observation: Observation, zone: Zone) -> bool:
        point = observation.bbox.normalized_centroid(
            width=self.scene.metadata.width,
            height=self.scene.metadata.height,
        )
        return zone.contains(point)

    @staticmethod
    def _contiguous_groups(
        observations: Iterable[Observation],
    ) -> tuple[tuple[Observation, ...], ...]:
        groups: list[list[Observation]] = []
        for observation in observations:
            if not groups or observation.frame_index != groups[-1][-1].frame_index + 1:
                groups.append([observation])
            else:
                groups[-1].append(observation)
        return tuple(tuple(group) for group in groups)

    def _to_zone_visit(
        self,
        track: Track,
        zone: Zone,
        observations: tuple[Observation, ...],
    ) -> ZoneVisit:
        first, last = observations[0], observations[-1]
        frame_period = 1.0 / self.scene.metadata.fps
        time_range = TimeRange(start_s=first.timestamp_s, end_s=last.timestamp_s)
        evidence = EvidenceRef(
            scene_id=self.scene.metadata.scene_id,
            track_ids=(track.track_id,),
            frame_indices=tuple(observation.frame_index for observation in observations),
            time_range=time_range,
            description=f"{track.concept} observed in zone {zone.label}",
        )
        return ZoneVisit(
            track_id=track.track_id,
            zone_id=zone.zone_id,
            time_range=time_range,
            observed_duration_s=(last.timestamp_s - first.timestamp_s) + frame_period,
            evidence=evidence,
        )

    def _observation_evidence(
        self,
        track: Track,
        observation: Observation,
        description: str,
    ) -> EvidenceRef:
        timestamp = observation.timestamp_s
        return EvidenceRef(
            scene_id=self.scene.metadata.scene_id,
            track_ids=(track.track_id,),
            frame_indices=(observation.frame_index,),
            time_range=TimeRange(start_s=timestamp, end_s=timestamp),
            description=f"{description} for {track.concept}",
        )
