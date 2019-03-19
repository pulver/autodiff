#! /bin/bash
#
# Copyright 2017 James E. King, III
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
#      http://www.boost.org/LICENSE_1_0.txt)
#
# Bash script to run in travis to perform a Coverity Scan build
#

#
# Environment Variables
#
# COVERITY_SCAN_NOTIFICATION_EMAIL  - email address to notify
# COVERITY_SCAN_TOKEN               - the Coverity Scan token (should be secure)
# SELF                              - the boost libs directory name

set -ex

pushd /tmp
rm -rf coverity_tool.tgz cov-analysis*
wget -nv https://scan.coverity.com/download/linux64 --post-data "token=$COVERITY_SCAN_TOKEN&project=Autodiff" -O coverity_tool.tgz
tar xzf coverity_tool.tgz
COVBIN=$(echo $(pwd)/cov-analysis*/bin)
export PATH=$COVBIN:$PATH
popd
rm -rf $BOOST_ROOT/libs/$SELF/coverity-user-config.jam
echo "using gcc : 7.4 ; " > $BOOST_ROOT/libs/$SELF/coverity-user-config.jam
ci/travis/build.sh clean --user-config=$BOOST_ROOT/libs/$SELF/coverity-user-config.jam
rm -rf cov-int/
cov-build --dir cov-int ci/travis/build.sh --user-config=$BOOST_ROOT/libs/$SELF/coverity-user-config.jam
tar cJf cov-int.tar.xz cov-int/
curl --form token="$COVERITY_SCAN_TOKEN" \
     --form email="$COVERITY_SCAN_NOTIFICATION_EMAIL" \
     --form file=@cov-int.tar.xz \
     --form version="$(git describe --tags)" \
     --form description="Autodiff" \
     https://scan.coverity.com/builds?project="Autodiff"
