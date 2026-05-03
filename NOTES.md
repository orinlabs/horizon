Ok here's how you build an eval. We'll use a simple synthetic example:

*"A teammate asks a follow-up question in today's inbox. The answer was learned in a prior session trace, not in the current environment. Does the agent retrieve that prior detail and write the right reply?"*

1. **Identify your trace**. The trace should contain the prior-session information the agent needs later.
2. **Find your cutoff point**. Pick the moment where the trace contains the needed learning, but the current environment still has live work pending.
3. **Build your environment**. Each eval has a trace and an environment. Once the trace is ingested, the agent runs in an environment and needs to do something. The environment should contain tools that are stateful, so that when the agent interacts with the environment, the state change can be seen. The tools should mirror the trace - things like inboxes, documents, or tasks. The whole idea is that we give the agent a task, view the environment state after the agent is done, and see if it succeeded based on the state.
4. **Verify**. Make sure there are no possible answers to this eval that the agent could do that would be marked wrong but could actually be correct. We want to make that false negative rate as low as possible.

Eval design notes:

1. **Preserve trace shape.** Even synthetic examples should look like the message/tool-call history an agent would actually see.
2. **The cutoff matters as much as the failure case.** A good cutoff is a moment where the trace contains the needed learning, the current environment has live work pending, and the agent must decide what to do without being directly asked the answer.
3. **Seed environment state from the post-cutoff task.** The trace given to the agent ends at the cutoff, but the eval environment should contain the current live request that makes the old information useful.
4. **The tool surface must mirror the trace.** If the trace shows tools named `sms_list`, `task_list`, `show_account`, `create_document`, etc., the agent should be able to call those names directly. API agents should receive those as native function tools, and shell/installed agents should get command shims that route to the same stateful backend. Otherwise the eval measures tool-interface mismatch instead of continual learning.
5. **The CLI is just the state boundary.** A stateful CLI like `acadia` is useful as the backing implementation, but it should not force the agent to behave like a shell agent. Route native model tool calls through the CLI under the hood.
6. **Verifier should score writes, not thoughts.** For a plan-level eval, inspect final state: newly created/updated documents, tasks, profile notes, goals, and outbound SMS. Do not rely on the agent's reasoning or final chat response.
7. **Diff updated documents carefully.** If the agent updates a long existing document, the judge should see a real before/after diff, not the whole final document. Otherwise old pre-existing content can falsely count as something the agent produced.
8. **Prefer a strict LLM judge over brittle proxies.** For plan-quality evals, the judge should decide whether the agent produced a concrete, executable activity, using structured outputs with a strict JSON schema. Simple keyword checks are useful for debugging but should not be the scoring rule.
9. **Plan-only evals are useful but limited.** They test whether the agent retrieves prior learning into planning context. They do not test whether the agent follows through during a live session. A higher-fidelity version would add an LLM student simulator and score the session transcript.
10. **A "too easy" eval is still informative at first.** If large-context or strong retrieval harnesses pass, that is a valid signal. If everyone passes, increase difficulty by requiring multiple specific goals, extending the trace, or moving from plan-only to live-session simulation.
11. **If an agent fails, that's ok!** We are building evals that current models are pretty bad at.

