import os, numpy, time, sys

sizes = [100000, 1000000, 10000000]

n = 10
now = time.strftime("%c")
start_dir = os.getcwd()

d = '.'
tests = [(o, os.path.join(d,o)) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

for o in tests:
    os.chdir(o[1])
    compile_string = "futhark-opencl -o {0}.bin ./{0}.fut".format(o[0])

    os.system(compile_string)

    for s in sizes:
        input_file = "data/{0}_size_{1}.input".format(o[0], s)
        output_file = "data/{0}_size_{1}.output".format(o[0], s)
        os.system("touch temp_time")
        os.system("touch temp_res")
        print "./{0}.bin -t temp_time -r {1} < {2} > ./temp_res".format(o[0], n, input_file)

        os.system("./{0}.bin -t temp_time -r {1} < {2} > ./temp_res".format(o[0], n, input_file))

        with open("./temp_res", "r") as res:
            with open(output_file, "r") as output:
                result_list = [int(i.replace("i32", "")) for i in res.read().strip("[]\n").split(",")]
                output_list = [int(i) for i in output.read().strip("[]\n").split(",")]
                if result_list != output_list:
                    print "Wrong result on test {}.".format(o[0])
                    print result_list[0:10]
                    print output_list[0:10]
                    
        with open("./temp_time", "r") as time:
            with open("results/times.txt", "a") as record:
                new_times = [int(t) for t in time.read().split()]
                if new_times is not []:
                    record.write("Time: %s\n" % now)
                    record.write("Size: %d\n" % s)
                    record.write("Repetitions: %d\n"%n)
                    record.write("Mean: %f\n" % (sum(new_times)/float(n)))
                    for t in new_times:
                        record.write("%d\n"%t)
                    record.write("=========================\n")
        os.system("rm temp_time temp_res")
    os.chdir(start_dir)


