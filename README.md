# TimbreLab

A four-person M112 project for exploring music and comparing machine learning models.

Start with [the team brief](docs/team-brief.md) for module ownership, shared interfaces and acceptance checks.

## First integration check

Open data/examples/audio_fixture.csv. Each row is one clip. clip_id and recording_id are identifiers, genre is the label, and the other columns are numeric predictors. These invented values are only for testing the interface. Use real labelled recordings to evaluate models.

data/examples/regression_fixture.csv contains invented bills, party sizes and tips for testing numeric target selection.

## Contributing

Clone this repository. Create a branch for one task, commit a small working change, push the branch and open a pull request. Ask the teammate who uses your output to review it. Keep full datasets and generated models out of Git.
