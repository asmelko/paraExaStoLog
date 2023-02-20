from exastolog import *
from os.path import dirname, join
import glob

for f in glob.glob("data/database-big/*"):
    model = Model(f)