# -*- coding: utf-8 -*-"""-------------------------------------------------   File Name：     LZW   Description :   Author :       付学明   date：          2018/5/23-------------------------------------------------   Change Activity:                   2018/5/23:-------------------------------------------------__author__ = '付学明'"""#inputs = "ABBABABACD",dict = {'A':1,'B':2,'C':3,'D':4}def LZW(inputs = "AAAA",dict = {'A':1,'B':2,'C':3,'D':4}):	prefix = ''	output=list('')	length=len(inputs)	i=0	for nextchar in inputs:		curStr = prefix + nextchar		i=i+1		if curStr in dict:			prefix=curStr			if i == length :				output.append(dict[prefix])		else:			output.append(dict[prefix])			dict[curStr] = len(dict) + 1			prefix = nextchar			if i == length:				output.append(dict[prefix])			#if dict[prefix] not in set(output):							print(dict)	print(output)	outputcode=''	for i in output:		outputcode = outputcode + str(i)+ ' '	#print(outputcode)	return outputcode	def deLZW(inputs='1 2 2 4 7 3',dict = {1:'A',2:'B',3:'C',4:'D'}):		inputlist = inputs.strip(' ').split(' ')	#print(inputlist )	inputarray = [int(i) for i in inputlist]	oldCodeStr = ''	dictsize=len(dict)	ans = ''	for curCode in inputarray:		if curCode > len(dict)+1:			print("error")			break		if curCode in dict:			curCodeStr = dict[curCode]					else:			curCodeStr = oldCodeStr + oldCodeStr[0]				if oldCodeStr != '':			dictsize = dictsize + 1			dict[dictsize] = oldCodeStr +curCodeStr[0]							oldCodeStr = curCodeStr		ans = ans +curCodeStr	print(dict)	print(ans)	if __name__ == "__main__":	code = LZW()	deLZW(inputs='1 5 1',)	#'1 2 4 8 1'