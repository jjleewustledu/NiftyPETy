#!/data/nil-bluearc/raichle/jjlee/anaconda2/envs/nipet/bin/python

# From https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys
import getopt
import respet.recon


def main(argv):
   usage = 'godo.py <tracer_rawdata_location> <t0> <t1> <frame#> <umap_index>'
   try:
      opts, args = getopt.getopt(argv, "h", ["help"])
   except getopt.GetoptError:
      print usage
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print usage
         sys.exit()
   print 'godo received ' + str(argv)
   recon.Reconstruction.godo(argv[0], argv[1], argv[2], argv[3], argv[4])


if __name__ == "__main__":
   main(sys.argv[1:])
