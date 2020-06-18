#!/usr/bin/env perl

sub version_number {
    @a = split(m%\.%, $_[0] );
    return($a[0]*1000000+$a[1]*1000+$a[2]*1);
}

if  (scalar @ARGV != 2) {
  print("$0 available supported");
  exit(1);
}

$av = $ARGV[0];
$max_av=version_number($av);
$sp = $ARGV[1];
$max_sp=version_number($sp);

#print "$max_av / $max_sp\n";
if ($max_av < $max_sp) {
    print "$av";
} else {
    print "$sp";
}

exit(0);
