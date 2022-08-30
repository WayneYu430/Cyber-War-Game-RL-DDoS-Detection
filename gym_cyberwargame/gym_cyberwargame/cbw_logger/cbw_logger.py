import logging
import os.path
from logging import handlers


class CBW_Logger():
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='debug', fmt='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s', output=True):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        # print(self.logger.level)
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        # sh.setLevel(self.level_relations.get(level))
        th = handlers.RotatingFileHandler(filename=filename, maxBytes=10 * 1024 * 1024, backupCount=5,
                                          encoding="utf-8")  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        th.setFormatter(format_str)  # 设置文件里写入的格式
        # th.setLevel(self.level_relations.get(level))
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        if output:  # 是否输出屏幕
            self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


if __name__ == '__main__':
    # os.path.join("./cbw_logger/test.log")
    log = CBW_Logger('test.log', level='info')
    testList = [1, 2, 3]
    host_info = {'1': 12, '2': 12}
    # log.logger.debug(testList)
    # log.logger.debug(host_info)
    log.logger.debug("get info{}".format(host_info))
    # log.logger.warning('警告')
    # log.logger.error('报错')
    # log.logger.critical('严重')
    # CBW_Logger('error.log', level='error').logger.error('error')
