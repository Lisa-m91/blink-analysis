import numpy as np
from functools import partial
from pickle import dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from blink_analysis.analyse import *

def test_main(tmpdir, capsys):
    np.random.seed(4)

    rois_f = tmpdir.join('rois.pickle')
    rois = np.random.normal(size=(2, 10, 9, 9))
    with rois_f.open("wb") as f:
        for roi in rois:
            dump(roi, f)

    ons_f = tmpdir.join('on.pickle')
    ons = np.random.normal(size=(2, 10)) > 0
    with ons_f.open("wb") as f:
        for on in ons:
            dump(on, f)

    main([str(rois_f), str(ons_f)])
