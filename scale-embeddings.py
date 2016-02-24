#!/usr/bin/python

"""
found on http://metaoptimize.com/projects/wordreprs/ and adapted

"""

words = []
values = []
import sys,string, numpy
for l in sys.stdin:
	d = string.split(l)
	#words.append(d[0])
	values.append([float(x) for x in d[0:]])

values = numpy.array(values)
values /= numpy.std(values)
values *= 0.1

for i in range(len(values)):
	#print words[i],
	for v in values[i]:
		print v,
	print
