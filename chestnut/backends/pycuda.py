from pycuda.gpuarray import GPUArray as DeviceArray
import codepy.cgen as c
import pycuda.curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

#class cuda_context:
#    def __enter__(self):
#        import pycuda.driver
#        from pycuda.tools import make_default_context
#
#        pycuda.driver.init()
#
#        self.context = make_default_context()
#        self.device = self.context.get_device()
#        return self.context
#
#    def __exit__(self, type, value, traceback):
#        print "context pop"
#        self.context.pop()
#        print "context nulling"
#        del self.context
#
#        from pycuda.tools import clear_context_caches
#        print "cache clearing"
#        clear_context_caches()
#        print "done"
