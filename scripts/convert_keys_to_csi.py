import sys
from wn_utils import WN_Utils

model_outputs_path = sys.argv[1]

if '.key.' in model_outputs_path:
    model_outputs_csi_path = model_outputs_path.replace('.key.', '.csi.key.')
elif model_outputs_path.endswith('.key'):
    model_outputs_csi_path = model_outputs_path.replace('.key', '.csi.key')

wn_utils = WN_Utils()

print('Writing %s ...' % model_outputs_csi_path)
with open(model_outputs_csi_path, 'w') as outputs_converted_f:
    with open(model_outputs_path) as outputs_f:
        for line in outputs_f:
            elems = line.strip().split()
            inst_id, inst_sk = elems[0], elems[1:]

            # only matters for converting eval sets
            inst_sk = inst_sk[0]

            inst_csi = wn_utils.sk2csi(inst_sk)

            if inst_csi is None:
                inst_csi = 'OTHER'

            outputs_converted_f.write('%s %s\n' % (inst_id, inst_csi))
