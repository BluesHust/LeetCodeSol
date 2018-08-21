class TwoKeysSolution:
    def minSteps(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return 0
        memory = [0 for i in range(n + 1)]
        return self._minSteps(memory, n)

    def _minSteps(self, memory, n):
        if n == 1 or memory[n] != 0:
            return memory[n]
        for i in range(n-1,int(n**0.5)-1, -1):
            if n % i == 0:
                memory[n] = self._minSteps(memory, i) + n // i
                return memory[n]
        #n只能分解为1*n
        memory[n]=n
        return memory[n]
    def run(self):
        print(self.minSteps(18))


class Solution:
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        n=len(nums)
        dp=[[0 for i in range(2001)] for i in range(n)] #注意数组的每行大小
        #填第0行
        dp[0][1000+nums[0]]=1
        dp[0][1000-nums[0]]=1
        for i in range(1,n):
            for sum in range(-1000,1001):
                if dp[i-1][sum+1000]>0: #前面的情况存在才修改后面的情况
                    dp[i][sum+nums[i]+1000]+=dp[i-1][sum+1000]
                    dp[i][sum-nums[i]+1000]+=dp[i-1][sum+1000]
        for i in range(n):
            for j in range(len(dp[0])):
                if dp[i][j]>0:
                    print("{} {} {}".format(i,j,dp[i][j]))
        return dp[n-1][S+1000]

    def run(self):
        nums=[1,1,1,1,1]
        print(self.findTargetSumWays(nums,3))

def main():
    sol=Solution()
    sol.run()

if __name__ == '__main__':
    main()