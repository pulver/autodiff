#! /bin/bash
#
# Copyright 2017 - 2019 James E. King III
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
#      http://www.boost.org/LICENSE_1_0.txt)
#
# Bash script to run in travis to perform codecov.io integration
#

# assumes cwd is the top level directory of the boost project
# assumes an environment variable $SELF is the boost project name

set -ex

. ci/travis/enforce.sh

if [ -z "$GCOV" ]; then
    if [ "${B2_TOOLSET%%-*}" == "gcc" ]; then
        ver="${B2_TOOLSET#*-}"
        GCOV=gcov-${ver}
    else
        GCOV=gcov-7 # default
    fi
fi

B2_VARIANT=debug
ci/travis/build.sh cxxflags=-fprofile-arcs cxxflags=-ftest-coverage linkflags=-fprofile-arcs linkflags=-ftest-coverage

# switch back to the original source code directory
cd $TRAVIS_BUILD_DIR

# get the version of lcov
lcov --version

# coverage files are in ../../b2 from this location
lcov --gcov-tool=$GCOV --base-directory "$BOOST_ROOT/libs/$SELF" --directory "$BOOST_ROOT" --capture --output-file all.info

# all.info contains all the coverage info for all projects - limit to ours
# first we extract the interesting headers for our project then we use that list to extract the right things
for f in `for f in include/boost/*; do echo $f; done | cut -f2- -d/`; do echo "*/$f*"; done > /tmp/interesting
echo headers that matter:
cat /tmp/interesting
xargs -L 999999 -a /tmp/interesting lcov --gcov-tool=$GCOV --extract all.info {} "*/libs/$SELF/src/*" --output-file coverage.info

# dump a summary on the console - helps us identify problems in pathing
lcov --gcov-tool=$GCOV --list coverage.info

#
# upload to codecov.io
#
curl -s https://codecov.io/bash > .codecov
chmod +x .codecov
./.codecov -f coverage.info -X gcov -x "$GCOV"
