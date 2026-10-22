#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Generic sensitive-pattern scan for this repository (run before every PR / release).
# It contains no real secrets: only regular expressions and an allow-list of known-harmless matches.
#
# Usage: bash scripts/dev/scan_sensitive.sh [REPO_DIR]     (REPO_DIR defaults to the repository root)
#        SHOW=1 bash scripts/dev/scan_sensitive.sh          (also print every reported match)
# Requires: bash, find, perl 5 (present on macOS and Ubuntu by default).
# Exit codes: 0 no findings, 1 findings, 2 usage error.
#
# Scanned: every text file under REPO_DIR except .git/ and this script itself (its regex definitions would
# match themselves). Binary files (perl -B) are skipped; the repo's binaries are the test media in assets/.
#
# Patterns (name: what it looks for):
#   users_path      macOS home paths            /Users/<name>
#   home_non_ubuntu Linux home dirs             /home/<name> other than /home/ubuntu (the DLAMI default user)
#   instance_id     EC2 instance IDs            i-<8..17 hex>
#   resource_id     AWS resource IDs            ami- / sg- / subnet- / vpc- / cr- + <8..17 hex>
#   account_id      12-digit AWS account IDs    a standalone run of exactly 12 digits, not touching a `.`, so the
#                                               fractional part of a float such as 6256.403503502952 is not reported
#   pem_path        SSH key paths               <path>.pem
#   private_ipv4    RFC 1918 addresses          10.x.x.x, 172.16-31.x.x, 192.168.x.x
#   private_host    EC2 private DNS names       ip-10-… / ip-172-… / ip-192-… host names
#   public_ipv4     any other dotted quad       x.x.x.x that is not private and not in the allow-list below
#   aws_access_key  AWS access key IDs          AKIA / ASIA + 16 upper-case alphanumerics
#
# A dotted quad must not be part of a longer dotted number, so version strings with more than four parts
# (2.11.0.1.0.1284, 0.24.0.1.1.0) never match, and parts above 255 (2.27.5334.0) are not IPv4; four-part
# versions that look like an IP are allow-listed one by one.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REPO=${1:-$ROOT}
[ -d "$REPO" ] || { echo "no such dir: $REPO" >&2; exit 2; }
SELF=$(basename "$0")

find "$REPO" -path "$REPO/.git" -prune -o -type f ! -name "$SELF" -print0 |
  REPO="$REPO" SHOW="${SHOW:-0}" perl -0 -e '
use strict; use warnings;

my $O = qr/(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])/;
my @patterns = (
  [users_path      => qr{/Users/[A-Za-z0-9._-]+}],
  [home_non_ubuntu => qr{/home/[A-Za-z0-9._-]+}],
  [instance_id     => qr{(?<![A-Za-z0-9_-])i-[0-9a-f]{8,17}(?![A-Za-z0-9_-])}],
  [resource_id     => qr{(?<![A-Za-z0-9_-])(?:ami|sg|subnet|vpc|cr)-[0-9a-f]{8,17}(?![A-Za-z0-9_-])}],
  [account_id      => qr{(?<![0-9.])[0-9]{12}(?![0-9.])}],
  [pem_path        => qr{[A-Za-z0-9_./~-]+\.pem(?![A-Za-z0-9])}],
  [private_ipv4    => qr{(?<![0-9.])(?:10\.$O|172\.(?:1[6-9]|2[0-9]|3[01])|192\.168)\.$O\.$O(?![0-9.])}],
  [private_host    => qr{(?<![A-Za-z0-9_-])ip-(?:10|172|192)-[0-9]{1,3}-[0-9]{1,3}-[0-9]{1,3}}],
  [public_ipv4     => qr{(?<![0-9.])$O\.$O\.$O\.$O(?![0-9.])}],
  [aws_access_key  => qr{(?:AKIA|ASIA)[0-9A-Z]{16}}],
);
my $private = qr{^(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)};

# Allow-list: exact token per pattern, each with the reason it is harmless.
my %allow = (
  home_non_ubuntu => {
    "/home/ubuntu" => "DLAMI default user on the instance; not personal",
  },
  public_ipv4 => {
    "127.0.0.1" => "loopback",
    "0.0.0.0"   => "wildcard bind address",
    "2.34.10.0" => "aws-neuronx-runtime-lib / collectives version, not an IP",
    "2.30.2.0"  => "aws-neuronx-dkms version, not an IP",
    "2.32.28.0" => "aws-neuronx-tools version (README version notes), not an IP",
  },
  account_id => {
    "302795194368" => "element count of the FP8 checkpoint (results/conversion/fp8_convert_check_fulldir.json)",
    "349634457472" => "byte size (total_size) of the FP8 checkpoint in the same file",
  },
);

my $repo = $ENV{REPO};
my %count = map { $_->[0] => 0 } @patterns;
my @report;
$/ = "\0";
my @files = <STDIN>;
chomp @files;
$/ = "\n";
for my $f (sort @files) {
  next if -B $f;
  open(my $fh, "<", $f) or next;
  (my $rel = $f) =~ s{^\Q$repo\E/}{};
  while (my $line = <$fh>) {
    for my $p (@patterns) {
      my ($name, $re) = @$p;
      while ($line =~ /$re/g) {
        my $tok = $&;
        next if $name eq "public_ipv4" && $tok =~ $private;
        next if exists $allow{$name} && exists $allow{$name}{$tok};
        $count{$name}++;
        push @report, "  $name  $rel:$.: $tok";
      }
    }
  }
  close $fh;
}
my $total = 0;
for my $p (@patterns) {
  printf "%-16s %d\n", $p->[0], $count{$p->[0]};
  $total += $count{$p->[0]};
}
print "$_\n" for ($ENV{SHOW} eq "1" ? @report : ());
print "total_findings=$total\n";
exit($total ? 1 : 0);
'
