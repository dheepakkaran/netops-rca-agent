# RB-04 Packet loss on several access routers at once
Symptoms: %QOS-4-DROP loss alarms on several access routers at the same time.
Check: whether the affected routers share one uplink router. If they do, the shared parent is the likely cause even if its own metrics look calm: check its links, CPU and BGP sessions (see RB-01, RB-02, RB-03).
Next step: investigate the shared parent first; per-router fixes on the access layer waste time.
