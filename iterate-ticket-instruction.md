# Workflow: Iterate a Ticket

## Research Context
This repository contains machine learning research code. It consists of data components, data processing files, and model training files. Before starting a ticket, familiarize yourself with the data directories, data splits, and training scripts so the user does not have to specify their locations every time.

## Ticket Structure
Tickets provided by the user will generally contain:
1. **Why** we need to do this.
2. **What** to do.

## Execution Workflow
The goal is to make iterative steps. Focus on small changes rather than completing the entire ticket in one massive change.

1. **Read and Analyze:** Read the ticket provided by the user.
2. **Determine Next Step:** Decide the next logical step to make a one-step progress based on the "What" list (these do not need to be addressed in strict order). If this is not the first iteration of the ticket, continue the iteration based on user feedback from the previous iteration.
3. **Clarify:** Ask the user for clarification if any requirements are ambiguous.
4. **Plan:** If the requirements are clarified, present a brief plan for this specific next step to the user and ask if this plan looks good.
5. **Implement:** If the user is happy with the plan, write the code for the change following the `AGENTS.md` guidelines.
6. **Iterate:** Repeat steps 4 and 5 until the iteration (or ticket) is complete.
7. **Review:** Make a summary of the changes.
8. **Verify:** ask the user to review and comment.
