import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio

import desc.io
from desc.plotting import (
    plot_grid,
    plot_3d
)

folder = sys.argv[1]
path = os.path.join(folder, "eq_result.h5")
eq = desc.io.load(path)

fig = plot_3d(eq, "|B|")
fig.write_html(folder + "/result_3d.html")
pio.write_image(fig, folder + "/result_3d.png")
