# Check what branches exist locally
git branch

# If no branches exist, you need to create one and make an initial commit
# Create and switch to a new branch called main
git checkout -b main

# Add your README file
git add README.md

# Make your first commit
git commit -m "Add README.md with project description"

# Push to GitHub, setting the upstream branch
git push -u origin main