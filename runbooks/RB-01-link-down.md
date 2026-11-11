# RB-01 Uplink down
Symptoms: on the root router, traffic falls to near zero and packet loss on its uplink jumps; a %LINK-3-UPDOWN "changed state to down" alarm names the interface toward the parent. Routers behind it show packet loss and higher probe latency a few seconds later.
Check: interface counters and optics on the named port, the far-end port on the parent, recent maintenance tickets.
Next step: open a ticket for the root router's uplink; do not reroute traffic from the downstream routers, they are symptoms.
