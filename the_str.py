#3.1 验证回文串
class ValidPalindrome:

    def valid(self, s):
        '''
        :type s:str
        :param s:
        :return:
        '''
        s=s.lower()
        n=len(s)
        if n==0: #空字符串定义为回文串
            return True
        left,right=0,n-1
        while left<right:
            if not s[right].isalpha() and not  s[right].isdigit():
                right-=1
                continue
            if not s[left].isalpha() and not  s[left].isdigit():
                left+=1
                continue
            if s[left]!=s[right]:
                return False
            left+=1
            right-=1
        return True

    def run(self):
        s="A man, a plan, a canal: Panama"
        print(self.valid(s))


#3.2 实现strStr()
class ImplementstrStr:

    def strStr(self,haystack,needle):
        '''
        :type haystack:str
        :type needle:str
        :param haystack:
        :param needle:
        :return:
        '''
        n=len(needle)
        if n==0:
            return 0
        for i in range(len(haystack)-n+1):
            k=0
            # while haystack[i+k]==needle[k] and k<n: 这里错了，要先保证下标不越界
            while k<n-1 and haystack[i+k]==needle[k]:
                k+=1
            #if k==n-1:
            if k==n:
                return i
        return -1


#3.3 string to integer (atoi)
class StringToInteger:


    #我的错误解法，判断过于复杂
    # 并且未能有效判断数据的大小
    def myAtoi(self,s):
        '''
        :type s:str
        :param s:
        :return:
        '''
        n=len(s)
        symbol=['+','-']
        if n<1:return 0 #空白字符串，不转换
        cur=0
        start,end=None,None
        while cur<n:
            if not s[cur].isdigit():
                if start==None: #如果还未截取到任何有用字符
                    if s[cur].isspace():
                        cur+=1
                        continue
                    elif s[cur] in symbol:
                        start=cur
                        cur+=1
                        continue
                    else:
                        return 0 #开头既不是数字也不是符号和空白
                else:
                    end=cur
                    break
            else:
                cur+=1
        if cur==n and start!=None:
            end=cur
        res=int(s[start:end])
        return res

    def atoi(self,s):
        '''
        :type s:str
        :param s:
        :return:
        '''
        import sys
        INT_MAX=sys.maxsize  #注意，不同的机器上可能不同
        INT_MIN=-(INT_MAX+1)
        if s==None:return 0
        num=0 #记录提取出的数字
        sign=1 #num的符号
        n=len(s)
        i=0
        #跳过所有空白符
        while i<n and s[i]==' ':
            i+=1
        #跳过正负号
        if i<n and s[i]=='+':
            i+=1
        elif i<n and s[i]=='-':
            sign=-1
            i+=1
        #开始处理数字
        for j in range(i,n):
            if s[j]<'0' or s[j]>'9': #遇到数字以外的字符则停止处理
                break
            if num>INT_MAX//10 or \
                (num==INT_MAX//10 and int(s[j])>INT_MAX%10):
                return INT_MAX if sign==1 else INT_MIN
            num=num*10+int(s[j])
        return num*sign

    def run(self):
        s="-91283472332"
        print(self.atoi(s))

#3.5 最长回文子串
class LongestPalindromicSubstring(object):

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        n=len(s)
        cache=[[False for j in range(n)]for i in range(n)]
        max_len=1
        start=0
        #下面开始填充cache，从列开始填充
        for col in range(n):
            cache[col][col]=True
            for row in range(col):
                if s[col]==s[row] and (col-row==1 or cache[row+1][col-1]):
                    cache[row][col]=True
                    if col-row+1>max_len:
                        max_len=col-row+1
                        start=row
        return s[start:start+max_len]

    def manacher(self,s):
        s1="$#"+"#".join(s)+"#%"
        n=len(s)
        max_half=0
        mid=0
        for i in range(n):
            j=1
            while i-j>=0 and i+j<n and s1[i-j]==s1[i+j]:
                j+=1
            if j-1>max_half:
                max_half=j-1
                mid=i
        return s[(mid-max_half)//2:(mid+max_half)//2+1]


    def run(self):
        s="babaa"
        print(self.longestPalindrome(s))


#3.8 最长公共前缀
class LongestCommonPrefix:

    def longestCommonPrefix(self, strs):
        '''
        :type strs: List[str]
        :rtype: str
        :param strs:
        :return:
        '''
        if len(strs) == 0:
            return ''
        m = len(strs[0])
        # 纵向扫描
        for index in range(m):
            for s in strs:
                if index + 1 > len(s) or s[index] != strs[0][index]:
                    return strs[0][0:index]
        return strs[0]


    def longestCommonPrefix2(self,strs):

        """
        纵向扫描
        :type strs: List[str]
        :rtype: str
        """
        if len(strs)==0:
            return ''
        max_idx = len(strs[0]) - 1
        for s in strs:
            for idx in range(max_idx + 1):
                if idx + 1 > len(s) or s[idx] != strs[0][idx]:
                    max_idx = idx - 1
                    break
        return strs[0][0:max_idx + 1]

    def run(self):
        strs=["flower","flow","flight"]
        print(self.longestCommonPrefix2(strs))

#3.12 报数
class CountAndSay:

    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        cache = [None for i in range(n + 1)]
        cache[1] = "1"
        for i in range(1, n):
            count = 1
            num = cache[i][0]
            nextNum = []
            for j in range(1, len(cache[i])):
                if cache[i][j] == num:
                    count += 1
                else:
                    nextNum.append(str(count))
                    nextNum.append(num)
                    count = 1
                    num = cache[i][j]
            nextNum.append(str(count))
            nextNum.append(num)
            cache[i + 1] = "".join(nextNum)
        return cache[n]


    def countAndSay_refine(self,n):
        '''

        :param n:
        :return:str
        '''
        say='1'
        for i in range(0,n-1):
            next=[]
            count=1
            num=say[0]
            for j in range(1,len(say)):
                if say[j]!=num:
                    next.append(str(count))
                    next.append(num)
                    count=1
                    num=say[j]
                else:
                    count+=1
            next.append(str(count))
            next.append(num)
            say="".join(next)
        return say


    def run(self):
        n=input("输入n\n")
        print(self.countAndSay_refine(int(n)))

#3.13 找出字符串数组中的所有异位构词
class Anagrams:

    def anagrams(self,strs):
        '''
        :type strs:list[str]
        :param strs:
        :return: list[list]
        '''
        from collections import defaultdict
        wordmap=defaultdict(list)
        for s in strs:
            wordmap[str(sorted(s))].append(s)
        return wordmap.values()

    def run(self):
        strs=["eat","tea","tan","ate","nat","bat"]
        print(self.anagrams(strs))

class SimplifyPath:

    def _splitStr(self,string,sep=" "):
        '''
        :type string:str
        :param string:
        :param sep: str
        :return:list
        '''
        res=[]
        subStr=[]
        for s in string:
            if s!=sep:
                subStr.append(s)
            else:
                res.append("".join(subStr))
                subStr=[]
        res.append(''.join(subStr))
        return res

    def _removeEleFromList(self,theList,target):
        '''
        删除theList中的target，返回一个新列表
        :param theList:list
        :param target:
        :return:list
        '''
        next=0
        for i in range(len(theList)):
            if theList[i]!=target:
                theList[next]=theList[i]
                next+=1
        return theList[:next]

    def simplyPath(self,path):
        '''
        :type path:str
        :param path:
        :return: str
        '''
        pathStack=[]
        curDir=[]
        paths=path.split(sep='/')
        for d in paths:
            if d=='' or d=='.':
                continue
            elif d=='..' :
                if len(pathStack)>0:
                    pathStack.pop()
            else:
                pathStack.append(curDir)
        return '/'+'/'.join(pathStack)



    def run(self):
        path="/home//lgp"
        print(path.split(sep='/'))
        print(self._removeEleFromList(self._splitStr(path,sep='/'),target=''))



def main():
    sol=SimplifyPath()
    sol.run()

if __name__=="__main__":
    main()

