#encoding: utf-8

import sys

from math import log

''' usage:
	python tfidftrain.py train_data.txt model.txt
'''

def cleanlist(lin):

	rs = []
	for lu in lin:
		if lu:
			rs.append(lu)

	return rs

def calculateidf(featc, ndata):

	rsd = {}
	_nd = float(ndata)
	for k, v in featc.items():
		rsd[k] = log(ndata / float(v))

	return rsd

def calculatetf(featfd, classd):

	rsd = {}
	for clas, fd in featfd.items():
		_nfeatc = float(classd[clas])
		tmp = {}
		for ft, freq in fd.items():
			tmp[ft] = float(freq) / _nfeatc
		rsd[clas] = tmp

	return rsd

def buildtfidf(tfd, idfd, build_unk=True, avg_unkv=False, min_unkv=False, unk_percent=0.001, unk_max=16):

	rsd = {}
	for clas, fd in tfd.items():
		tmp = {}
		if build_unk:
			if avg_unkv:
				tfidfl = []
			elif min_unkv:
				m_unkv = 0.0
		for ft, tf in fd.items():
			_tfidf = tf * idfd[ft]
			tmp[ft] = _tfidf
			if build_unk:
				if avg_unkv:
					tfidfl.append(_tfidf)
				elif min_unkv and _tfidf < m_unkv:
					m_unkv = _tfidf
		if build_unk:
			if avg_unkv:
				ncount = max(min(len(tfidfl) * unk_percent, unk_max), 1)
				tfidfl.sort()
				unk_tfidf = 0.0
				for _tfidf in tfidfl[:ncount]:
					unk_tfidf += _tfidf
				tmp["<unk>"] = unk_tfidf / ncount
			elif min_unkv:
				tmp["<unk>"] = m_unkv
			else:
				tmp["<unk>"] = 0.0
		rsd[clas] = tmp

	return rsd

def norm(cld, nd):

	rsd = {}
	_nd = float(nd)
	for k, v in cld.items():
		rsd[k] = float(v) / _nd

	return rsd

# srcf: source file to train the model with format: _class_label _feature1 _feature2 ... _featuren
# modelf: file to save the trained model
# norm_feat: if a feature repeated several times in a line, reduce its frequency to 1 in that line
# norm_bias: normalize bias to a probability distribution

def handle(srcf, modelf, norm_feat=False, norm_bias=True):

	feat_freq = {}
	class_freq = {}

	ndata = 0
	nfeat = 0
	feat_n = {}
	class_n = {}

	# count
	with open(srcf, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = cleanlist(tmp.decode("utf-8").split())
				_clas, _features = tmp[0], tmp[1:]
				if norm_feat:
					_features = list(set(_features))
				for _f in _features:
					if _clas in feat_freq:
						if _f in feat_freq[_clas]:
							feat_freq[_clas][_f] += 1
						else:
							feat_freq[_clas][_f] = 1
					else:
						feat_freq[_clas] = {_f:1}
					feat_n[_f] = feat_n.get(_f, 0) + 1
				class_freq[_clas] = class_freq.get(_clas, 0) + 1
				class_n[_clas] = class_n.get(_clas, 0) + len(_features)
				nfeat += len(_features)
				ndata += 1

	# calculate IDF
	feat_n = calculateidf(feat_n, nfeat)
	# calculate TF
	feat_freq = calculatetf(feat_freq, class_n)

	# merge TFIDF
	model = buildtfidf(feat_freq, feat_n)

	if norm_bias:
		class_n = norm(class_freq, ndata)

	ens = "\n".encode("utf-8")
	with open(modelf, "wb") as f:
		f.write(repr(model).encode("utf-8"))
		f.write(ens)
		f.write(repr(class_n).encode("utf-8"))
		f.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
