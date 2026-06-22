package TensorRT::Edge::LLM;

use strict;
use warnings;

our $VERSION = '0.01';

1;
__END__

=head1 NAME

TensorRT::Edge::LLM - Low-level Perl XS bindings for TensorRT-Edge-LLM BatchManager

=head1 DESCRIPTION

This module provides the low-level, high-performance Perl XS bindings for the
TensorRT-Edge-LLM runtime (BatchManager).

It is used as the foundational engine wrapper by high-level serving abstractions
such as C9h::LLM.

=cut
