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

sub __clean { 
    my $t = $_[0]; 
    $t=~s%^\s+%%; 
    $t=~s%\s+$%%; 
    return($t); 
}

sub get_cto { 
    my ($x) = ($_[0] =~ m%^\|\s+([^\s]+)\s+\|%);
    my @t = (split(m%[_-]%, $x));
#    print scalar @t, " == ", join(" : ", @t), "\n";
    return(&__clean($t[0]), &__clean($t[1]), &__clean($t[2]), &__clean($t[3]) )
        if (scalar @t == 4);
    return(999, &__clean($t[0]), &__clean($t[1]), &__clean($t[2]) )
        if (scalar @t == 3);
    error_quit("Unable to extract CTO version from \"".$_[0]."\"");
}

sub comp_v { 
    my @x=split(m%\.%, $_[0]); 
    $x[1] = 0
        if (scalar @x == 1);
    $x[2] = 0
        if (scalar @x == 2);
    my $v=1000000*$x[0]+1000*$x[1]+$x[2]; 
#    print("\n$v  "); 
    return($v); 
}

sub __sort {
    my @a_cto = &get_cto($a);
#    print("\n%\%a%".join("/", @a_cto));
    my @b_cto = &get_cto($b);
#    print("\n%\%b%".join("/", @b_cto));
    return (
            (&comp_v($a_cto[0]) <=> &comp_v($b_cto[0]))
            || (&comp_v($a_cto[1]) <=> &comp_v($b_cto[1]))
            || (&comp_v($a_cto[2]) <=> &comp_v($b_cto[2])
            || $a_cto[3] <=> $b_cto[3] )
    );
}

my @sorted = sort __sort @lines;
print join("\n", @sorted) . "\n";
