#!/bin/bash

directory="./"

phrase="Workforce shortages could therefore increase pressures on inflation."

grep -rl "$phrase" "$directory"/*.txt


