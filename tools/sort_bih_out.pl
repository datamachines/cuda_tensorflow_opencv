#!/usr/bin/env perl

use strict;

sub error_quit {
    print "ERROR: " . join("", @_);
    exit(1);
}

if (scalar @ARGV != 1) {
    print "Usage: $0 infile"
}

my $infile = $ARGV[0];

open FILE, "<$infile"
    or error_quit("Problem reading $infile");
chomp(my @lines = <FILE>);
close FILE;

sub get_cto { my @x = ($_[0] =~ m%^\|.+?\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|%); return(@x); }
sub comp_v { my @x=split(m%\.%, $_[0]); return(1000000*$x[0]+1000*$x[1]+$x[2]); }
sub __sort {
    my @a_cto = &get_cto($a);
    my @b_cto = &get_cto($b);
    return (
            (&comp_v($a_cto[0]) <=> &comp_v($b_cto[0]))
            || (&comp_v($a_cto[1]) <=> &comp_v($b_cto[1]))
            || (&comp_v($a_cto[2]) <=> &comp_v($b_cto[2]))
    );
}

my @sorted = sort __sort @lines;
print join("\n", @sorted) . "\n";
