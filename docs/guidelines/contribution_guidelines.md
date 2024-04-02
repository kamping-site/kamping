Contributing to KaMPIng {#contribution_guidelines}
==================================================

How to report a bug
-------------------
- Before anything else, please make sure you are on the latest version, the bug you are experiencing may have been fixed already!
- Use the search function to see if someone else has already submitted the same bug report.
- Try to describe the problem with as much detail as possible.
- Some bugs may only occur on certain machines or implementations/versions of MPI. Please add information about your hardware (Infiniband, Omnipath, Ehternet, ...) and MPI implementation (Intel MPI, OpenMPI, MPICH). Please specify the version of your MPI implementation and KaMPIng.
- If possible, add instructions on how to reproduce the bug.
- Please use the following **[template](https://github.com/kamping-site/kamping/issues/new?assignees=&labels=Type%3A+Possible+bug&template=bug_report.yml)**.

How to submit a feature request
-------------------------------
- Make sure you are using the latest version. Perhaps the feature you are looking for has already been implemented.
- Use the search function to see if someone else has already submitted the same feature request. If there is another request already, please upvote the first post instead of commenting something like "I also want this".
- To make it easier for us to keep track of requests, please only make one feature request per issue.
- Give a brief explanation about the problem that may currently exist and how your requested feature solves this problem.
- Try to be as specific as possible. Please not only explain what the feature does, but also how.
- Please use the following **[template](https://github.com/kamping-site/kamping/issues/new?assignees=&labels=&template=feature_request.yml)**.

Submit a pull request
---------------------
- If you want to work on a feature that has been requested or fix a bug that has been reported on the "issues" page, add a comment to it so that other people know that you are working on it.
- Fork the repository.
- If your pull request fixes a bug that has been reported or implements a feature that has been requested in another issue, try to mention it in the message, so that it can be closed once your pull request has been merged. If you use special keywords in the [commit comment](https://help.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue) or [pull request text](https://github.blog/2013-05-14-closing-issues-via-pull-requests/), GitHub will close the issue(s) automatically.
- Always add extensive unit tests for your pull request and make sure that they pass.
- Please do not upgrade dependencies or build tools unless you have a good reason for it. Doing so can easily introduce bugs that are hard to track down.
- If you plan to do a change that touches many files (10+), please ask beforehand. This usually causes merge conflicts for other developers.
- Please read our [coding guidelines](https://github.com/kamping-site/kamping/blob/main/docs/coding_guidelines.md)
- Please read our [documentation guidelines](https://github.com/kamping-site/kamping/blob/main/docs/documentation_guidelines.md)
- Please read our [testing guidelines](https://github.com/kamping-site/kamping/blob/main/docs/testing_guidelines.md)
- Commit only corrections of typos and similar minor fixes directly to the `main` branch. For everything else, use `feature-` and `fix-` branches and merge them to the `main` branch using a Pull Request (PR).
- Use a draft PR while you are still working on your pull request. Once you are ready for review, convert it into a regular PR.
- Label your Pull Request with one or more of these labels:
  - `discussion` if you are looking for comments but your PR is not ready for a full review yet.
  - `dependent` if your PR is dependent on another issue/PR/discussion.
- Each Pull Request has to be reviewed by at least one person who is not the author of the code. Everyone involved in a discussion, including the pull request's author, can close a discussion once its matter is resolved. Avoid writing "Done" etc. when resolving a discussion, as this generates a lot of low-entropy mails; simply close the discussion if the matter is resolved completely. If unsure, leave the discussion open and ask the reviewer if the change is sufficient. Resolving a discussion might for example include moving the discussion to a new issue or implementing the requested changes. Once all discussions are closed, all CI checks are successful, and there are no more rejecting reviews, it is the pull request's author's responsibility to merge the changes into the main branch. Use squash and merge to merge a pull request back into the main branch.
- For pull requests by contributers without the necessary access rights, a reviewer will perform the merge.

Building From Source
--------------------------
```shell
# Don't forget the --recursive flag
git clone --recursive https://github.com/kamping-site/kamping.git
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build

# Alternatively, if you have at least CMake 3.20
git clone --recursive https://github.com/kamping-site/kamping.git
# Supported build types are release relwithdeb debug
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

