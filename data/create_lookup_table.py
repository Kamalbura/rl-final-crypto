import pandas as pd
import numpy as np
from datetime import datetime
import json

# Create the complete 30-state lookup table as Excel file
def create_lookup_table_excel():
    """
    Creates an Excel file with the complete 30-state lookup table
    for battery-optimized cryptographic algorithm selection.
    """
    
    # Define the lookup table data
    lookup_data = [
        # <20% Battery Level (States 1-6)
        (1, 0, 0, 0, 'RANDOM_PRE', 5.8, 'Pre-quantum', 'Critical battery, no threat - random pre-quantum'),
        (2, 0, 0, 1, 'RANDOM_PRE', 5.8, 'Pre-quantum', 'Critical battery, important mission but no threat'),
        (3, 0, 1, 0, 'KYBER', 6.2, 'Post-quantum', 'Critical battery but threat present - min PQC power'),
        (4, 0, 1, 1, 'KYBER', 6.2, 'Post-quantum', 'Critical battery, important mission, threat present'),
        (5, 0, 2, 0, 'FALCON', 7.1, 'Post-quantum', 'Critical battery but confirmed threat - security override'),
        (6, 0, 2, 1, 'FALCON', 7.1, 'Post-quantum', 'Critical battery, important mission, confirmed threat'),
        
        # 20-40% Battery Level (States 7-12)
        (7, 1, 0, 0, 'KYBER', 6.2, 'Post-quantum', 'Low battery, normal threat - efficient PQC'),
        (8, 1, 0, 1, 'KYBER', 6.2, 'Post-quantum', 'Low battery, important mission, normal threat'),
        (9, 1, 1, 0, 'KYBER', 6.2, 'Post-quantum', 'Low battery, confirming threat - stay efficient'),
        (10, 1, 1, 1, 'DILITHIUM', 6.5, 'Post-quantum', 'Low battery, important mission upgrade'),
        (11, 1, 2, 0, 'FALCON', 7.1, 'Post-quantum', 'Low battery but confirmed threat - security override'),
        (12, 1, 2, 1, 'FALCON', 7.1, 'Post-quantum', 'Low battery, important mission, confirmed threat'),
        
        # 40-60% Battery Level (States 13-18)
        (13, 2, 0, 0, 'DILITHIUM', 6.5, 'Post-quantum', 'Medium battery, normal threat - balanced choice'),
        (14, 2, 0, 1, 'DILITHIUM', 6.5, 'Post-quantum', 'Medium battery, important mission, normal threat'),
        (15, 2, 1, 0, 'DILITHIUM', 6.5, 'Post-quantum', 'Medium battery, confirming threat - maintain balance'),
        (16, 2, 1, 1, 'SPHINCS', 6.8, 'Post-quantum', 'Medium battery, important mission upgrade'),
        (17, 2, 2, 0, 'FALCON', 7.1, 'Post-quantum', 'Medium battery, confirmed threat - security override'),
        (18, 2, 2, 1, 'FALCON', 7.1, 'Post-quantum', 'Medium battery, important mission, confirmed threat'),
        
        # 60-80% Battery Level (States 19-24)
        (19, 3, 0, 0, 'SPHINCS', 6.8, 'Post-quantum', 'Good battery, normal threat - high security available'),
        (20, 3, 0, 1, 'SPHINCS', 6.8, 'Post-quantum', 'Good battery, important mission, normal threat'),
        (21, 3, 1, 0, 'SPHINCS', 6.8, 'Post-quantum', 'Good battery, confirming threat - high security'),
        (22, 3, 1, 1, 'FALCON', 7.1, 'Post-quantum', 'Good battery, important mission upgrade'),
        (23, 3, 2, 0, 'FALCON', 7.1, 'Post-quantum', 'Good battery, confirmed threat - max security'),
        (24, 3, 2, 1, 'FALCON', 7.1, 'Post-quantum', 'Good battery, important mission, confirmed threat'),
        
        # 80-100% Battery Level (States 25-30)
        (25, 4, 0, 0, 'FALCON', 7.1, 'Post-quantum', 'High battery, normal threat - max security available'),
        (26, 4, 0, 1, 'FALCON', 7.1, 'Post-quantum', 'High battery, important mission, normal threat'),
        (27, 4, 1, 0, 'FALCON', 7.1, 'Post-quantum', 'High battery, confirming threat - max security'),
        (28, 4, 1, 1, 'FALCON', 7.1, 'Post-quantum', 'High battery, important mission, confirming threat'),
        (29, 4, 2, 0, 'FALCON', 7.1, 'Post-quantum', 'High battery, confirmed threat - max security'),
        (30, 4, 2, 1, 'FALCON', 7.1, 'Post-quantum', 'High battery, important mission, confirmed threat'),
    ]
    
    # Create DataFrame
    df = pd.DataFrame(lookup_data, columns=[
        'State_ID', 'Battery_Level', 'Threat_Status', 'Mission_Criticality',
        'Algorithm', 'Power_W', 'Type', 'Decision_Logic'
    ])
    
    # Add human-readable columns
    battery_map = {0: '<20%', 1: '20-40%', 2: '40-60%', 3: '60-80%', 4: '80-100%'}
    threat_map = {0: 'Normal', 1: 'Confirming', 2: 'Confirmed'}
    mission_map = {0: 'Routine', 1: 'Important'}
    
    df['Battery_Range'] = df['Battery_Level'].map(battery_map)
    df['Threat_Level'] = df['Threat_Status'].map(threat_map)
    df['Mission_Type'] = df['Mission_Criticality'].map(mission_map)
    
    # Reorder columns for better readability
    df = df[['State_ID', 'Battery_Level', 'Battery_Range', 'Threat_Status', 'Threat_Level',
             'Mission_Criticality', 'Mission_Type', 'Algorithm', 'Power_W', 'Type', 'Decision_Logic']]
    
    # Save to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f'c:/Users/burak/Desktop/rl-final-crypto/data/lookup_table_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main lookup table
        df.to_excel(writer, sheet_name='Lookup_Table', index=False)
        
        # Algorithm statistics
        algo_stats = df['Algorithm'].value_counts().reset_index()
        algo_stats.columns = ['Algorithm', 'Count']
        algo_stats['Percentage'] = (algo_stats['Count'] / len(df) * 100).round(1)
        algo_stats.to_excel(writer, sheet_name='Algorithm_Stats', index=False)
        
        # Power consumption analysis
        power_stats = df.groupby('Power_W')['Algorithm'].count().reset_index()
        power_stats.columns = ['Power_W', 'State_Count']
        power_stats['Percentage'] = (power_stats['State_Count'] / len(df) * 100).round(1)
        power_stats.to_excel(writer, sheet_name='Power_Analysis', index=False)
        
        # State space mapping
        state_mapping = {
            'battery_levels': list(battery_map.values()),
            'threat_levels': list(threat_map.values()),
            'mission_types': list(mission_map.values()),
            'algorithms': df['Algorithm'].unique().tolist(),
            'total_states': len(df)
        }
        
        # Convert dict to DataFrame for Excel
        mapping_df = pd.DataFrame([(k, str(v)) for k, v in state_mapping.items()], 
                                  columns=['Parameter', 'Values'])
        mapping_df.to_excel(writer, sheet_name='State_Mapping', index=False)
    
    print(f"✅ Excel lookup table created: {excel_path}")
    
    # Also save as JSON for Python use
    json_path = 'c:/Users/burak/Desktop/rl-final-crypto/data/state_mappings.json'
    with open(json_path, 'w') as f:
        json.dump({
            'lookup_table': df.to_dict('records'),
            'state_mapping': state_mapping,
            'battery_map': battery_map,
            'threat_map': threat_map,
            'mission_map': mission_map,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"✅ JSON mappings created: {json_path}")
    
    return df, excel_path, json_path

if __name__ == "__main__":
    df, excel_path, json_path = create_lookup_table_excel()
    print(f"\n📊 Lookup Table Summary:")
    print(f"Total States: {len(df)}")
    print(f"Algorithms: {df['Algorithm'].nunique()}")
    print(f"Power Range: {df['Power_W'].min()}W - {df['Power_W'].max()}W")
    print("\n🎯 Files Created:")
    print(f"- Excel: {excel_path}")
    print(f"- JSON: {json_path}")
