use strict;
use warnings;
use ExtUtils::MakeMaker;
use Config;
use Cwd qw(abs_path);
use File::Basename qw(dirname);

my $PREFIX = $ENV{HOME} . '/perl5';

my $arch = `uname -m`;
chomp $arch;

my $CUDA_ROOT;
my $LIB_ARCH_DIR;
my $INC_ARCH_DIR;

if ($arch eq 'aarch64') {
    $CUDA_ROOT = '/usr/local/cuda/targets/sbsa-linux';
    $LIB_ARCH_DIR = '/usr/lib/aarch64-linux-gnu';
    $INC_ARCH_DIR = '/usr/include/aarch64-linux-gnu';
} else {
    $CUDA_ROOT = '/usr/local/cuda';
    $LIB_ARCH_DIR = '/usr/lib/x86_64-linux-gnu';
    $INC_ARCH_DIR = '/usr/include/x86_64-linux-gnu';
}

my $CUDA_INCLUDE = "$CUDA_ROOT/include";
my $CUDA_LIB = "$CUDA_ROOT/lib";
my $CUDA_LIB64 = "/usr/local/cuda/lib64";

my $TRT_EDGE_LLM_SOURCE = "..";
my $TRT_EDGE_LLM_BUILD = "$TRT_EDGE_LLM_SOURCE/build";
my $JSON_INCLUDE = "$TRT_EDGE_LLM_SOURCE/3rdParty/nlohmannJson/include";

# Path to prebuilt CuTe DSL FMHA artifacts
my $FMHA_LIB = "$TRT_EDGE_LLM_SOURCE/cpp/kernels/contextAttentionKernels/cuteDSLArtifact/$arch/libcutedsl_fmha_$arch.a";

# Detect Coral Edge TPU SDK
my $has_coral = 0;
if (-f "/usr/include/edgetpu.h" || -f "/usr/include/edgetpu/edgetpu.h") {
    print "Detected Coral Edge TPU SDK headers.\n";
    $has_coral = 1;
}

my $inc = "-I. -I$TRT_EDGE_LLM_SOURCE/cpp -I$INC_ARCH_DIR -I$CUDA_INCLUDE -I$JSON_INCLUDE -I$PREFIX/include";
# The XS modules will link ONLY against our shared bridge library.
my $curdir = abs_path(".");
my $libs = "-L$curdir -lbridge";
my $define = "";

if ($has_coral) {
    $inc .= " -I/usr/include/edgetpu";
    $define .= " -DHAS_CORAL";
}

# Find all plugin sources to bundle into libbridge.so
my @plugin_srcs = (
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/attentionPlugin/attentionPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/qwen3DeltaAttentionPlugin/qwen3DeltaAttentionPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/qwen3DeltaAttentionPlugin/qwen3DeltaAttention.cu",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/vitAttentionPlugin/vitAttentionPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/int4GroupwiseGemmPlugin/int4GroupwiseGemmPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/mamba/causalConv1dPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/mamba/mambaPlugin.cpp",
    "$TRT_EDGE_LLM_SOURCE/cpp/plugins/int4MoePlugin/int4MoePlugin.cpp",
);
my $plugin_src_str = join(' ', @plugin_srcs);

WriteMakefile(
    NAME             => 'TensorRT::Edge::LLM',
    AUTHOR           => 'C.J. Collier <cjac@google.com>',
    VERSION_FROM     => 'lib/TensorRT/Edge/LLM.pm',
    ABSTRACT_FROM    => 'lib/TensorRT/Edge/LLM.pm',
    LICENSE          => 'artistic_2',
    PL_FILES         => {},
    MIN_PERL_VERSION => '5.014',
    CONFIGURE_REQUIRES => {
        'ExtUtils::MakeMaker' => '0',
    },
    TEST_REQUIRES => {
        'Test::More'       => '0',
        'Test::Exception'  => '0',
    },
    PREREQ_PM => {
        'Moo'                       => '0',
        'Log::Any'                  => '0',
        'Types::Standard'           => '0',
    },
    # XS Configuration
    CC               => 'g++',
    LD               => 'g++',
    XSOPT            => '-C++',
    XSMULTI          => 1,
    TYPEMAPS         => ['typemap'],
    INC              => $inc,
    LIBS             => [$libs],
    DEFINE           => $define,
    MYEXTLIB         => 'libbridge.so',
    
    dist  => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
    clean => { FILES => 'TensorRT-Edge-LLM-* bridge.o libbridge.so libbridge.a lib/TensorRT/Edge/LLM/*.c' },
);

package MY;

sub test {
    my $self = shift;
    return <<'END_MAKE';
test ::
	@echo "Running tests..."
	LD_LIBRARY_PATH=. prove -It/lib -b -v t/*.t
END_MAKE
}

sub postamble {
    my $self = shift;
    
    return <<"END_MAKE";
\$(MYEXTLIB): cpp/bridge.cpp cpp/bridge.h
	\@mkdir -p \$(\@D)
	/usr/local/cuda/bin/nvcc -shared -o libbridge.so cpp/bridge.cpp $plugin_src_str \\
	   -Xcompiler -fPIC \$(INC) -I\$(PERL_INC) \\
	   -L$TRT_EDGE_LLM_BUILD/cpp -L$TRT_EDGE_LLM_BUILD/lib -L$PREFIX/lib -L$CUDA_LIB -L$CUDA_LIB64 -L$LIB_ARCH_DIR \\
	   -Xlinker --allow-multiple-definition -Xlinker -soname,libbridge.so \\
	   -ledgellmBuilder -ledgellmCore -ledgellmTokenizer $FMHA_LIB -lnvonnxparser -lnvinfer -lcuda -lcudart -lrt -lpthread -ldl

# Default to parallel build for all targets
MAKEFLAGS += -j\$(shell nproc)
END_MAKE
}
