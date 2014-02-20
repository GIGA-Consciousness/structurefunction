import os.path as op

from coma.interfaces import CMR_glucose

from coma.datasets import sample
data_path = sample.data_path()

in_file = op.join(data_path, 'data', 'Bend1',
                  'fswpQP733896-0002-00001-000001-01.nii.gz')


CMR_glucose(in_file, dose=8.15, weight=71,
            delay=63, glycemie=100, scan_time=15)
