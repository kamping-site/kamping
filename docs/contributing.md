Contributing to KaMPI.ng {#contribution_guidelines}
============

* Commit only corrections of typos and similar minor fixes directly to the `main` branch. For everything else, use `feature-` and `fix-` branches and merge them to the `main` branch using a Pull Request (PR).
* Label your Pull Request with one or more of these labels:
    * `wip` if you are still working on it.
    * `discussion` if you are looking for comments but your PR is not ready for a full review yet.
    * `dependent` if your PR is dependent on another issue/PR/discussion
    * `review` if your branch is ready for review.
* Each Pull Request has to be reviewed by at least one person who is not the author of the code. Everyone involved in a discussion, including the pull request's author, can close a discussion once its matter is resolved. Avoid writing "Done" etc. when resolving a discussion, as this generates a lot of low-entropy mails; simply close the discussion if the matter is resolved completely. If unsure, leave the discussion open and ask the reviewer if the change is sufficient. Resolving a discussion might for example include moving the discussion to a new issue or implementing the requested changes. Once all discussions are closed, all CI checks are successful, and there are no more rejecting reviews, it is the pull request's author's responsibility to merge the changes into the main branch. Use squash and merge to merge a pull request back into the main branch.
