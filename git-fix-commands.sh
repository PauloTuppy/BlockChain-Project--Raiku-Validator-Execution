# Check what branch you're on
git branch

# If no branch exists yet, create and switch to main branch
git checkout -b main

# Add your README file
git add README.md

# Commit the changes
git commit -m "Add README.md with project description"

# Push to GitHub, setting upstream tracking
git push -u origin main