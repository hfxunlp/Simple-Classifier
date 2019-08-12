#encoding: utf-8

import sys

''' usage:
	python predict.py text_data.txt model.txt test_result.txt bias_weight
'''

def loadmodel(mf):

	with open(mf, "rb") as frd:
		model = eval(frd.readline().strip().decode("utf-8"))
		bias = eval(frd.readline().strip().decode("utf-8"))

	return model, bias

def cleanlist(lin):

	rs = []
	for lu in lin:
		if lu:
			rs.append(lu)

	return rs

def predict(fl, model, bias=None, enable_unk=True, bias_weight=1.0):

	rsd = {}
	for clas, fd in model.items():
		_s = 0.0
		unkv = fd.get("<unk>", 0.0) if enable_unk else 0.0
		for ft in fl:
			_s += fd.get(ft, unkv)
		if bias is not None:
			_s += bias[clas] * bias_weight
		rsd[clas] = _s

	return rsd

# srcf: source file to predict
# modelf: file to load the model
# rsf: result file
# bias_weight: important hyper-parameter to tune, unless the model was trained with train.py
# enable_bias: enable bias during prediction
# enable_unk: enable <unk> tag during prediction
# norm_feat: if a feature repeated several times in a line, reduce its frequency to 1 in that line

def handle(srcf, modelf, rsf, bias_weight=1.0, enable_bias=True, enable_unk=True, norm_feat=False):

	model, bias = loadmodel(modelf)

	ens = "\n".encode("utf-8")
	with open(srcf, "rb") as frd, open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = cleanlist(line.strip().decode("utf-8").split())
			if norm_feat:
				tmp = list(set(tmp))
			tmp = predict(tmp, model, bias if enable_bias else None, enable_unk, bias_weight)
			fwrt.write(repr(tmp).encode("utf-8"))
			fwrt.write(ens)

if __name__ == "__main__":
	if len(sys.argv) > 4:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
	else:
		handle(sys.argv[1], sys.argv[2], sys.argv[3])
