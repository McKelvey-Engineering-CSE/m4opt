"""
Prepare a GW[...].fits FITS file for use with M4OPT
"""

from astropy.table import Table
from astropy.io import fits

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None, help="Input FITS file")

args = parser.parse_args()

fil = args.file

with fits.open(fil) as f:
    primary_hdu = f[0].copy()

    table = f[1]

    # understood to be the key for GPS time (Julian date)
    table.header["MJD-OBS"] = "2022-09-27 18:00:00.000"
    table.header["DATE-OBS"] = "2022-09-27 18:00:00.000"
    
    data = table.data
    cols = table.columns

    new_cols = fits.ColDefs([
        fits.Column(name=(c.name if c.name != 'T' else 'PROB'), format=c.format, array=data[c.name])
        for c in cols
    ])

    new_table = fits.BinTableHDU.from_columns(new_cols, header=table.header)

    new_hdul = fits.HDUList([primary_hdu, new_table])

    name_split = fil.split("/")
    name_split[-1] = name_split[-1].replace(".fits", "_CONVERTED.fits")

    new_hdul.writeto("/".join(name_split), overwrite=True)