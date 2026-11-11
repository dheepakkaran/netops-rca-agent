# RB-02 Control-plane CPU overload
Symptoms: CPU above 90% on one router with %SYS-4-CPUHOG naming a routing process; latency rises on that router first and, less, on routers that route through it. Traffic and loss stay close to normal.
Check: process list and the routing table churn on the root router, scheduled jobs, a recent software change.
Next step: ask the router owner to identify the process; consider rate-limiting control-plane traffic. Downstream latency clears when the root recovers.
