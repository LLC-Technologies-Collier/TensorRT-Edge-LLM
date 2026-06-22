#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
BEGIN { diag "@INC is: " . join(":", @INC); }
use AnyEvent;
use C9h::LLM;
BEGIN { diag "C9h::LLM loaded from: " . $INC{'C9h/LLM.pm'}; }
use TensorRT::Edge::LLM::Embedding;
use Log::Any::Adapter;
use Fcntl qw(:flock);

$ENV{EDGELLM_PLUGIN_PATH} = "../build/libNvInfer_edgellm_plugin.so";

# Use a lock file to prevent parallel tests from conflicting on the GPU
my $lock_file = "/tmp/c9h-llm-gpu-test.lock";
open my $lock_fh, '>', $lock_file or die "Could not open lock file $lock_file: $!";
diag "Waiting for GPU lock...";
flock($lock_fh, LOCK_EX) or die "Could not lock $lock_file: $!";
diag "GPU lock acquired.";

# Log::Any::Adapter->set('Stderr', log_level => 'debug');

my $engine_dir = "/srv/nfs/c9h-llm-data/qwen_1.5b_general_deploy/engine_110_new";

unless (-d $engine_dir) {
    plan skip_all => "Engine directory not found: $engine_dir";
}

subtest 'Embedding Extraction' => sub {
    unless (-d $engine_dir) {
        plan skip_all => "Engine directory not found: $engine_dir";
    }

    my $embedder = eval {
        TensorRT::Edge::LLM::Embedding->new(
            engine_dir => $engine_dir,
            edgellm_plugins => "../build/libNvInfer_edgellm_plugin.so",
        );
    };
    
    ok($embedder, "Created Embedding runner") or do {
        diag "Error: $@";
        plan skip_all => "Could not initialize Embedding runner (hardware/driver issue?)";
    };

    my $cv = AnyEvent->condvar;
    my $timer = AnyEvent->timer(after => 30, cb => sub { $cv->send("timeout") });
    
    my $vector;
    my $text = "The quick brown fox jumps over the lazy dog.";
    
    # We might need to run the extraction in a separate watcher or just use synchronous get_embedding
    # Since get_embedding is currently synchronous in the PM, we just wrap it.
    
    $vector = eval { $embedder->get_embedding($text) };
    $cv->send("done") unless $@;
    
    my $res = $cv->recv;
    is($res, "done", "Extraction finished without timeout");
    
    ok($vector, "Extracted embedding vector") or diag "Error: $@";
    is(ref($vector), 'ARRAY', "Vector is an array reference");
    
    # Qwen 1.5B hidden size is 1536
    is(scalar(@$vector), 1536, "Vector has correct dimension (1536)");
    
    note("First 5 components: " . join(", ", @$vector[0..4]));
    
    ok($vector->[0] != 0 || $vector->[1] != 0, "Vector is not all zeros");
};

done_testing();
