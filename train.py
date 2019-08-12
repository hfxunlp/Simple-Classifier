#encoding: utf-8

import sys

from math import sqrt

''' usage:
	python train.py train_data.txt init_model_from_tfidf.txt trained_model.txt train_epoch init_lr regularize_weight
'''

def loadmodel(mf):

	with open(mf, "rb") as frd:
		model = eval(frd.readline().strip().decode("utf-8"))
		bias = eval(frd.readline().strip().decode("utf-8"))

	return model, bias

def savemodel(mf, model, bias):

	ens = "\n".encode("utf-8")
	with open(mf, "wb") as fwrt:
		fwrt.write(repr(model).encode("utf-8"))
		fwrt.write(ens)
		fwrt.write(repr(bias).encode("utf-8"))
		fwrt.write(ens)

def cleanlist(lin):

	rs = []
	for lu in lin:
		if lu:
			rs.append(lu)

	return rs

def get_loss(fl, gold_class, model, bias=None, enable_unk=True, hi_strict=True):

	gold_s = 0.0
	fd = model[gold_class]
	unkv = fd.get("<unk>", 0.0) if enable_unk else 0.0
	for ft in fl:
		gold_s += fd.get(ft, unkv)
		if bias is not None:
			gold_s += bias[gold_class]

	err_class = []
	ms = gold_s
	for clas, fd in model.items():
		if clas != gold_class:
			_s = 0.0
			unkv = fd.get("<unk>", 0.0) if enable_unk else 0.0
			for ft in fl:
				_s += fd.get(ft, unkv)
			if bias is not None:
				_s += bias[clas]
			if _s >= ms:
				if hi_strict:
					err_class.append(clas)
				elif _s > ms:
					err_class = [clas]
					ms = _s

	return err_class

# srcf: training data
# modelf: file to load the model
# rsf_head: file head to save trained models, models will be saved to rsf_head_%epoch.txt
# train_epoch: number of epochs to train
# init_lr: initial learning rate
# regularize: weight for L2 regularization, 0.0 to disable
# quick_reduce_lr: reduce learning rate quickly
# enable_unk: enable unk for predicting
# strict: 0: only increase the score of gold class, 1: penalize the highest score class, 2: penalize all classes with scores higher than the gold class
# norm_feat: if a feature repeated several times in a line, reduce its frequency to 1 in that line

def handle(srcf, modelf, rsf, train_epoch=128, init_lr=0.1, regularize=0.0, quick_reduce_lr=0.7, enable_unk=True, strict=0, norm_feat=False):

	model, bias = loadmodel(modelf)

	lr = init_lr
	for i in range(1, train_epoch + 1):
		lr = (quick_reduce_lr * lr) if quick_reduce_lr > 0.0 else (init_lr / sqrt(i))
		nd = 0
		nerror = 0
		with open(srcf, "rb") as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = cleanlist(tmp.decode("utf-8").split())
					clas, feats = tmp[0], tmp[1:]
					if norm_feat:
						_features = list(set(_features))
					# forward
					err_clas = get_loss(feats, clas, model, bias, enable_unk, strict>1)
					# backward
					if err_clas:
						td = model[clas]
						for ft in feats:
							if ft in td:
								td[ft] += lr
							elif enable_unk:
								td["<unk>"] += lr
						bias[clas] += lr
						if strict > 0:
							for e_clas in err_clas:
								td = model[e_clas]
								for ft in feats:
									if ft in td:
										td[ft] -= lr
									elif enable_unk:
										td["<unk>"] -= lr
								bias[e_clas] -= lr
						# apply L2 regularization
						if regularize > 0.0:
							for clas, fd in model.items():
								for k, v in fd.items():
									fd[k] -= lr * v * regularize
							for clas, v in bias.items():
								bias[clas] -= lr * v * regularize
						nerror += 1
					nd += 1
		erate = float(nerror) / float(nd) * 100.0
		savemodel(rsf + "_%d_%.2f.txt" % (i, erate), model, bias)
		print("Epoch %d: lr %.3f, error rate %.2f" %(i, lr, erate))

if __name__ == "__main__":
	_larg = len(sys.argv)
	if _larg == 4:
		handle(sys.argv[1], sys.argv[2], sys.argv[3])
	elif _larg == 5:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
	elif _larg == 6:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]))
	elif _larg == 7:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
	else:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
