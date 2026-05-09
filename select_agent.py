from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import pandas as pd

# -------------------------- 配置项：替换成你的大模型信息 --------------------------
LLM_API_KEY = "替换成你的API密钥，比如sk-xxx"
LLM_BASE_URL = "替换成你的API地址，比如https://api.openai.com/v1 或者国内大模型的地址"
LLM_MODEL = "gpt-3.5-turbo" # 如果用国内模型改成对应模型名，比如qwen-max、glm-4

# -------------------------- 1. 定义各个功能工具：对应多Agent分工 --------------------------
# 工具1：多平台数据爬取（实际可以替换成对接平台开放API/真实爬取逻辑）
def crawl_ecomm_data(keyword: str) -> str:
    """爬取电商平台的品类数据、搜索热度数据"""
    # 模拟爬取结果，实际业务中替换为真实请求
    platform_data = {
        "平台": ["淘宝", "拼多多", "抖音电商"],
        "关键词": [keyword, keyword, keyword],
        "近7天搜索热度": [128000, 215000, 324000],
        "环比涨幅": ["+18%", "+32%", "+47%"],
        "主流均价区间": ["29-59元", "19-39元", "29-49元"]
    }
    df = pd.DataFrame(platform_data)
    return f"✅ 已完成{keyword}全平台基础数据爬取，结果如下：\n{df.to_markdown(index=False)}"

data_crawler_tool = Tool(
    name="多平台电商数据爬取",
    func=crawl_ecomm_data,
    description="爬取目标品类在淘宝/拼多多/抖音电商的搜索热度、价格、涨幅基础数据，输入是目标品类关键词，比如'便携式小风扇'"
)

# 工具2：竞品数据抓取分析
def get_competitor_analysis(keyword: str) -> str:
    """抓取目标品类的头部竞品的销量、评价、卖点数据"""
    # 模拟竞品分析结果，实际替换为真实爬取/平台API返回
    competitor_data = [
        {"竞品店铺层级": "头部大店", "月销": "1.2万+", "核心卖点": "低价走量、基础款", "用户差评痛点": "质量一般，做工粗糙，材质偏薄"},
        {"竞品店铺层级": "腰部店铺", "月销": "4500+", "核心卖点": "设计感强，细节做工好", "用户差评痛点": "价格偏高，发货慢，颜色色差大"}
    ]
    df = pd.DataFrame(competitor_data)
    return f"✅ 已完成{keyword}竞品分析，结果如下：\n{df.to_markdown(index=False)}"

competitor_tool = Tool(
    name="竞品信息分析",
    func=get_competitor_analysis,
    description="获取目标品类的头部竞品销量、核心卖点、用户差评数据，输入是目标品类关键词"
)

# 工具3：行业趋势分析（对接阿里指数/百度指数）
def analyze_trend(keyword: str) -> str:
    """分析品类的长期趋势，判断赛道潜力，挖掘细分需求"""
    # 模拟分析结果，实际替换为真实指数API返回
    trend_info = f"""
✅ 已完成{keyword}的趋势分析：
1. 长期趋势：近90天全网搜索热度持续上涨，即将进入夏季销售旺季，整体赛道仍在增长
2. 竞争格局：细分赛道暂无绝对头部垄断，中小卖家仍有切入机会
3. 用户未被满足需求：用户普遍关注『便携性』『性价比』『颜值』，目前市场缺口是「30-50元中价位带的品质款」
    """
    return trend_info

trend_analyze_tool = Tool(
    name="品类趋势分析",
    func=analyze_trend,
    description="分析目标品类的长期搜索趋势，挖掘用户未被满足的细分需求，输入是目标品类关键词"
)

# -------------------------- 2. 初始化多Agent系统 --------------------------
def main():
    # 初始化大模型
    llm = ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    # 注册所有工具
    tools = [data_crawler_tool, competitor_tool, trend_analyze_tool]

    # 初始化选品Agent：自动按照流程调用工具完成全链路分析
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True # 开启后会打印完整的思考过程，方便查看执行流程
    )

    # -------------------------- 3. 运行：输入目标品类，输出选品报告 --------------------------
    print("=== AI电商选品Agent启动 ===")
    target_category = input("请输入你要分析的品类关键词（比如：便携式手持小风扇）：")
    
    prompt = f"""
    我是电商卖家，需要你帮我分析{target_category}这个品类是否值得切入，请按照顺序完成以下步骤：
    1. 调用「多平台电商数据爬取」获取基础热度和价格数据
    2. 调用「竞品信息分析」获取竞品的优缺点和用户反馈
    3. 调用「品类趋势分析」获取赛道趋势和细分缺口
    最后整合所有信息，输出一份完整的选品报告，包含：赛道总结、竞品机会、切入建议、风险提示四个部分。
    """

    # 执行并输出结果
    result = agent.run(prompt)
    print("\n"+"="*50)
    print("📊 最终AI选品报告：\n")
    print(result)
    print("="*50)

if __name__ == "__main__":
    main()
