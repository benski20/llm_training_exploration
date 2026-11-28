"""Placement: GPU pods in the data centres nearest the users, started and stopped on demand
(docs/operations/placement.md).

Operator-side only (the client library never imports it; it verifies with :mod:`pgw.sdk.directory`):

* :mod:`pgw.placement.catalog`: data centres, metros, the RTT estimate between them, prices;
* :mod:`pgw.placement.predict`: decode tok/s per user from the RTT (the calibrated round-trip model);
* :mod:`pgw.placement.telemetry`: content-free demand reports, aggregated before anything is stored;
* :mod:`pgw.placement.forecast`: time-of-day profile and trend (a pod takes minutes to start);
* :mod:`pgw.placement.policy`: the controller (warm set, gain per $, hysteresis, idle timeout, spend caps,
  availability fallback) -> desired pods per data centre and model;
* :mod:`pgw.placement.publisher`: the signed endpoint directory;
* :mod:`pgw.placement.certs`: short-lived per-pod TLS certificates under the pinned provider CA;
* :mod:`pgw.placement.registry`: pod boot -> verify pack -> load -> self-tests -> health -> register;
* :mod:`pgw.placement.coldstart`: cold-start time components;
* :mod:`pgw.placement.packs`: per-data-centre pre-staged packs on network volumes, verified at boot.
"""
