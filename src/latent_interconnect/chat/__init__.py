"""Object-conditioned chat on the package's receivers and connectors (the last layer, beside ``run``).

``config`` (the deployment config), ``service`` (requests through the renderer, the write sources and the generation
backend's stream), ``encoding`` (live encoding by resident encoder workers), ``server`` (HTTP, SSE, auth), the static
UI and ``smoke`` (a deployment's smoke test over HTTP). The entry points: ``python -m latent_interconnect.chat``
(``__main__``) and the installed ``aimm-chat`` command (``cli``: defaults, connector selection, checks); the API
contract and how to run it: ``examples/guides/chat.md``.
"""
