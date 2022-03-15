for f in convDNN_intel_ufunc_cc  \
	convDNN_llvm_ufunc_cc  \
	convDNN_gnu_ufunc_cc \
	convDNN_intel_unrolled_cc  \
	convDNN_llvm_unrolled_cc  \
	convDNN_gnu_unrolled_cc \
	convDNN_llvm_polly_ufunc_cc \
	convDNN_llvm_polly_unrolled_cc\
	convDNN_intel_ufunc  \
	convDNN_llvm_ufunc  \
	convDNN_gnu_ufunc \
	convDNN_intel_unrolled  \
	convDNN_llvm_unrolled  \
	convDNN_gnu_unrolled \
	convDNN_llvm_polly_ufunc \
	convDNN_llvm_polly_unrolled
do
    ./${f} 1>> /dev/null 2>> data.csv
done
