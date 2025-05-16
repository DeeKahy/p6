import os
import re
from pathlib import Path

def generate_sequence():
    """Generate the complete sequence of interval-width values."""
    sequence = [
        "0.01", "0.005", "0.003", "0.002", "0.001",
        "0.0009", "0.0008", "0.0007", "0.0006", "0.0005"
    ]

    # Continue the sequence for future values
    current = "0.0005"

    # How many more values to generate
    num_additional = 30

    for _ in range(num_additional):
        # Split the current value
        parts = current.split('.')
        integer_part = parts[0]  # Will always be "0"
        decimal_part = parts[1]

        # Count leading zeros
        leading_zeros = 0
        for c in decimal_part:
            if c == '0':
                leading_zeros += 1
            else:
                break

        # Get significant digits
        significant_digits = decimal_part[leading_zeros:]
        value = int(significant_digits)

        # Determine next value
        if value == 1:
            # Add another decimal place
            leading_zeros += 1
            value = 9
        else:
            # Decrement
            value -= 1

        # Create new value and add to sequence
        new_value = f"{integer_part}.{'0' * leading_zeros}{value}"
        sequence.append(new_value)
        current = new_value

    return sequence

def get_existing_indices():
    """Get a list of existing file indices."""
    indices = []
    pattern = re.compile(r'fireflies-queries-(\d+)\.xml')

    for file in Path('.').glob('fireflies-queries-*.xml'):
        match = pattern.match(file.name)
        if match:
            indices.append(int(match.group(1)))

    return indices

def create_xml_file(index, interval_width):
    """Create or update the XML file with the given index and interval width."""
    filename = f"./fireflies-queries-{index}.xml"

    # XML template
    xml_content = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<property-set xmlns="http://tapaal.net/">
  
  <property>
    <id>Compute Euler number</id>
    <description>Compute Euler number</description>
    <smc confidence="0.95" interval-width="{interval_width}" time-bound="1000"/>
    <formula>
      <finally>
        <integer-eq>
          <tokens-count>
            <place>Euler_finished</place>
          </tokens-count>
          <integer-constant>1</integer-constant>
        </integer-eq>
      </finally>
    </formula>
  </property>
</property-set>
    '''

    # Write the file
    with open(filename, 'w') as f:
        f.write(xml_content)

    print(f"Created/updated file: {filename} with interval-width={interval_width}")

def update_existing_files_and_create_new():
    """Update existing files and create new ones based on the sequence."""
    # Generate the sequence
    sequence = generate_sequence()

    # Get existing indices
    existing_indices = get_existing_indices()

    # Sort by interval width (descending)
    index_mapping = {}

    for i, width in enumerate(sequence):
        index_mapping[width] = i

    # Update existing files first
    for xml_file in Path('.').glob('fireflies-queries-*.xml'):
        with open(xml_file, 'r') as f:
            content = f.read()

        # Extract current interval-width
        match = re.search(r'interval-width="([^"]+)"', content)
        if match:
            current_width = match.group(1)

            # Check if this width is in our sequence
            if current_width in index_mapping:
                # Get the correct index for this width
                correct_index = index_mapping[current_width]

                # Create or update the file with correct index
                create_xml_file(correct_index, current_width)

    # Create any missing files
    for i, width in enumerate(sequence):
        # Check if file exists for this index
        filename = f"./fireflies-queries-{i}.xml"
        if not os.path.exists(filename):
            create_xml_file(i, width)

def main():
    print("Starting to update XML files based on the sequence...")
    update_existing_files_and_create_new()
    print("Finished updating XML files.")

if __name__ == "__main__":
    main()

