# import pytest

# We don't want to be subject to the CASA dependency, so we need to create our own fake data.
# The best solution is probably just to read the visibilities out of the cube itself
# And store them online in an npz and then just download that each time we want to test.

# Regardless, we need a fixture which provides mock visibilities of the sort we'd
# expect from visread, but *without* the CASA dependency.
# Maybe we should just add a to_npz() method to visread.
# I think this is what we're going to have to do, unfortunately, since we're going to want to
# Write tutorials.

# pytest fixtures used in multiple files go here
# download the ALMA logo and produce a measurement set from it
# or just download the measurement set itself
# Create our own mock visibility cube using fake locations
