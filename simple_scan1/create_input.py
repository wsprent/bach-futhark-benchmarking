import sys, random, numpy

n = int(sys.argv[1])

# inp = [random.randint(0,100) for i in range(n)]
inp = numpy.random.randint(0, 100, size=n)
print "Created input list."
#plus10 = map(lambda x: x+10, inp)
map10 = numpy.vectorize(lambda x: x+10)
plus10 = map10(inp)
print "Completed map."
outp = numpy.cumsum(plus10)
print "Completed scan. Writing input."
with  open("data/simple_scan1_size_" + str(n) + ".input", "w") as f:
    f.write(str(inp.tolist()))
print "Wrote input. Writing output"
with  open("data/simple_scan1_size_" + str(n) + ".output", "w") as f:
    f.write(str(outp.tolist()))    
print "Done!"
