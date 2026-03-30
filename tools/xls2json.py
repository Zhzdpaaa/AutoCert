import pandas as pd
import json

def excel_awards_to_json(excel_path: str, json_path: str = None) -> dict:
    """
    将获奖情况Excel表格转换为JSON格式
    第一行：上传人信息
    第二行：表头
    第三行起：数据
    """
    # 读取Excel
    df = pd.read_excel(excel_path, header=None)
    
    # 获取上传人（第一行第二列）
    submitter = df.iloc[0, 1] if pd.notna(df.iloc[0, 1]) else None
    
    awards = []
    
    # 从第三行开始处理数据（索引2，跳过上传人行和表头行）
    for idx in range(2, len(df)):
        row = df.iloc[idx].tolist()
        
        # 跳过空行
        if pd.isna(row[0]) or str(row[0]).strip() == '':
            continue
        
        record = {
            'award_category': row[0] if pd.notna(row[0]) else None,
            'award_name': row[1] if pd.notna(row[1]) else None,
            'year': int(row[2]) if pd.notna(row[2]) else None,
            'level': row[3] if pd.notna(row[3]) else None,
            'grade': row[4] if pd.notna(row[4]) else None,
        }
        
        # 判断是个人还是团队
        team_rank = row[5]
        
        if team_rank == '个人':
            record['is_team'] = False
            record['team_rank'] = None
            record['bonus_points'] = float(row[6]) if pd.notna(row[6]) else None
            record['members'] = []
        else:
            record['is_team'] = True
            record['team_rank'] = int(team_rank) if pd.notna(team_rank) else None
            
            # 解析成员信息
            members = []
            col_idx = 6
            while col_idx < len(row) - 1:
                name = row[col_idx]
                score = row[col_idx + 1]
                
                if pd.notna(name) and str(name).strip():
                    members.append({
                        'name': str(name).strip(),
                        'points': float(score) if pd.notna(score) else None
                    })
                col_idx += 2
            
            record['members'] = members
        
        awards.append(record)
    
    # 最终结构
    result = {
        'student_name': submitter,
        'awards': awards
    }
    
    # 保存到JSON文件
    if json_path:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"JSON文件已保存到: {json_path}")
    
    return result


if __name__ == "__main__":
    excel_path = "/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/zuodong/2023-2024年获奖情况汇总.xlsx"
    json_path = "/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/zuodong/2023-2024年获奖情况汇总.json"
    
    data = excel_awards_to_json(excel_path, json_path)
    print(json.dumps(data, ensure_ascii=False, indent=2))
