package TensorRT::Edge::LLM::Builder;

use strict;
use warnings;
use Moo;
use Carp qw(croak);
use Log::Any qw($log);
use Types::Standard qw(Str Int Bool);
use Path::Tiny;

our $VERSION = '0.01';
use XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

sub init_plugins {
    my ($self, $plugin_path) = @_;
    $ENV{EDGELLM_PLUGIN_PATH} = $plugin_path;
    return $self->_xs_init_plugins($plugin_path);
}

sub get_runtime_version {
    my ($self) = @_;
    return $self->_xs_get_runtime_version();
}

sub build {
    my ($self, %args) = @_;
    my $onnx_dir   = $args{onnx_dir}   or croak 'onnx_dir required';
    my $engine_dir = $args{engine_dir} or croak 'engine_dir required';
    my $force      = $args{force}      // 0;
    
    my $max_input_len = $args{max_input_len} // 128;
    my $max_kv_cache  = $args{max_kv_cache}  // 4096;
    my $max_batch     = $args{max_batch_size} // 4;
    my $is_vlm        = $args{is_vlm} // 0;
    my $is_eagle      = $args{is_eagle} // 0;
    my $max_image_tokens = $args{max_image_tokens} // 256;
    my $weight_streaming_budget = $args{weight_streaming_budget} // -1;

    unless (-d $onnx_dir) {
        croak "onnx_dir '$onnx_dir' does not exist";
    }
    
    my $eng_path = path($engine_dir);
    if ($eng_path->child('config.json')->exists && $eng_path->child('llm.engine')->exists && !$force) {
        # Check if version matches
        my $config_path = $eng_path->child('config.json');
        $log->debugf("Checking engine version in %s", $config_path);
        eval {
            my $config_text = $config_path->slurp;
            if ($config_text =~ /"edgellm_version":\s*"([^"]+)"/) {
                my $model_ver = $1;
                my $runtime_ver = $self->get_runtime_version();
                $log->debugf("Version check: model=%s, runtime=%s", $model_ver, $runtime_ver);
                if ($model_ver ne $runtime_ver) {
                    $log->warnf("Engine version mismatch: model=%s, runtime=%s. FORCING REBUILD.", $model_ver, $runtime_ver);
                    $force = 1;
                }
            } else {
                $log->warnf("Could not find version in %s. FORCING REBUILD.", $config_path);
                $force = 1;
            }
        };
        if ($@) {
            $log->warnf("Error reading config.json: %s. FORCING REBUILD.", $@);
            $force = 1;
        }
        
        if (!$force) {
            $log->infof("Engine artifacts found in %s and version matches, skipping.", $engine_dir);
            return 1;
        }
    }

    $log->infof('Starting engine build from %s to %s', $onnx_dir, $engine_dir);
    $eng_path->mkpath;

    my $ok = $self->_xs_build(
        $onnx_dir, 
        $engine_dir, 
        $max_input_len, 
        $max_kv_cache, 
        $max_batch,
        $is_vlm,
        $weight_streaming_budget,
        $max_image_tokens,
        $is_eagle
    );

    if (!$ok) {
        croak "C++ LLMBuilder failed to build engine";
    }

    $log->infof('Engine build complete.');
    return 1;
}

1;
