# AUDL-Advanced-Stats

Retrieve, compile, and visualize advanced stats from the AUDL.

## Installation

To install it directly from Github, you can enter the following command into Anaconda Prompt or the command line:

```sh
pip install git+https://github.com/JohnLithio/AUDL-Advanced-Stats.git
```

If you already have it installed and wish to incorporate recent changes, run the following command in Anaconda Prompt or the command line:

```sh
pip install git+https://github.com/JohnLithio/AUDL-Advanced-Stats.git --upgrade
```

## Usage

Most of the heavy lifting is done by the `Game` and `Season` classes. These classes will retrieve the JSON data from the AUDL endpoint for each game and save it to a local directory. You can also configure an AWS bucket to store the data, but that's really not necessary.

Once the raw data is stored, the events for each team from each game can be processed and saved locally as well, along with some overall info about the league such as games, players, and teams.

Using this data, statistics for each game and season can be calculated and some other visualizations can be created.

See `Example.ipynb` for some examples of the usage and timing.
