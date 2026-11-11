# RB-03 BGP session flapping after a configuration change
Symptoms: repeated %BGP-5-ADJCHANGE Down / Up messages on one router, often with little change in its own CPU, traffic or loss. The routers that depend on it show packet loss and some latency because routes keep being withdrawn. The loudest routers are the victims, not the cause.
Check: the last configuration commit on the flapping router (timers, policy, neighbor addresses), the hold-timer, the peer's logs.
Next step: roll back the last BGP change on the flapping router after approval; confirm the session stays up for 10 minutes.
