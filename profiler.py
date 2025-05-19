import pstats
import glob

# Create a new combined profile file
combined = pstats.Stats()

# Find all profile files
for file in glob.glob("profile_*.prof"):
    combined.add(file)

# Save the merged profile
combined.dump_stats("combined_profile.prof")
print("Merged profiles into combined_profile.prof")