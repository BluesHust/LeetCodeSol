#树节点的定义
class TreeNode:
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None


#5.1.1 二叉树先序遍历
class PreorderTraversal:
    #递归实现
    def preorderTraversal_rec(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return: List[int]
        '''
        res=[]
        self._preorder(root,res)
        return res

    def _preorder(self,root,res):
        '''
        :type root:TreeNode
        :type res:list
        :param root:
        :param res:
        :return:
        '''
        if root==None:
            return
        res.append(root.val)
        self._preorder(root.left,res)
        self._preorder(root.right,res)

    #迭代实现
    def preorderTraversal(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return:
        '''
        res = []
        #:type list[node]
        s = []  # stack
        if root != None:
            s.append(root)
        while len(s) > 0:
            tmp = s.pop()
            res.append(tmp.val)

#5.1.2 中序遍历
class InorderTraversal:

    def inorderTraversal_rec(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return:
        '''
        res=[]
        self._inorderTravel(root,res)
        return res

    def _inorderTravel(self,root,res):
        '''
        :type root:TreeNode
        :type res:list
        :param root:
        :param res:
        :return:
        '''
        if root==None:
            return
        res.append(root.val)
        self._inorderTravel(root.left,res)
        self._inorderTravel(root.right,res)

    def inorderTravel(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return:
        '''
        res=[]
        s=[] #type:list[TreeNode]
        p=root #type:TreeNode
        # 访问完左子树和根结点后栈为空
        # 但此时还可能有右子树,因此需要
        # 加上条件p!=None
        while len(s)!=0 or p!=None:
            if p!=None:
                #可以继续向左走
                s.append(p)
                p=p.left
            else:
                #走到底了，回退,访问根节点
                p=s.pop()
                res.append(p.val) #访问该结点
                #如果右子树存在，还要访问右子树
                #如果右子树不存在，则会在下个
                # 循环进行回退
                p=p.right
        return res

#5.1.3 后序遍历
class PostTravel:

    #用两个栈实现，会遍历两遍
    def postTravel(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return:
        '''
        res=[]
        tmp=[] #type:list[TreeNode]
        postorder=[] #type:list[TreeNode]
        if root==None:
            return res
        tmp.append(root)
        while len(tmp)!=0:
            node=tmp.pop()
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
            postorder.append(node)
        while len(postorder)!=0:
            res.append(postorder.pop().val)
        return res

    def postTravel_leetcode(self,root):
        '''
        :type root:TreeNode
        :param root:
        :return: list[int]
        '''
        res = []  # type:list[int]
        node_stack = []  # type:list[TreeNode]
        if root == None:
            return res
        node_stack.append(root)
        p = root.left  # 表示当前结点
        while len(node_stack) != 0:
            # 先左走到底
            while p != None:
                node_stack.append(p)
                p = p.left
            #在向左走的过程中不会访问结点
            #所以要更新q的值
            q = None  #表示上次访问的结点
            while len(node_stack) != 0: #开始访问node_stack[-1]子树
                p = node_stack.pop()
                if p.right == q:  # 如果右子树为空或者已经访问过
                    res.append(p.val)
                    q = p  # 更新访问信息
                else:
                    # 还未访问过右子树，先处理右子树
                    node_stack.append(p) #先把根节点放回栈
                    p = p.right
                    break #在右子树上一直向左走到底
        return res


#5.1.4 层级遍历
class LevelOrderTraversal:

    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = list()
        self._levelorder(1, root, res)
        return res

    def _levelorder(self, level, root, res):
        '''
        :type level:int
        :type root:TreeNode
        :type res:list[list]
        :param level:表明根结点的层级，以便将结果保存到正确的位置
        '''
        if root == None:
            return
        if level > len(res):
            res.append(list())
        res[level - 1].append(root.val)
        self._levelorder(level + 1, root.left, res)
        self._levelorder(level + 1, root.right, res)


class SameTree:
    #递归实现
    def isSameTree_rec(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        elif p == None or q == None:
            return False
        return (p.val == q.val and
                self.isSameTree_rec(p.left, q.left) and
                self.isSameTree_rec(p.right, q.right))

    #迭代实现
    def isSameTree(self,p,q):
        '''
        :type p:TreeNode
        :type q:TreeNode
        :param p:
        :param q:
        :return: bool
        '''
        tree = []  # type:list[TreeNode]
        tree.append(p)
        tree.append(q)
        while len(tree) != 0:
            q = tree.pop()
            p = tree.pop()
            # 判断当前结点
            if q == None and p == None:
                continue
            elif q == None or p == None:
                return False
            if q.val != p.val:
                return False
            # 判断左右子树
            tree.append(p.left)
            tree.append(q.left)
            tree.append(p.right)
            tree.append(q.right)
        return True


    def run(self):
        tree1=TreeNode(1)
        tree1.left=TreeNode(2)
        tree1.right=TreeNode(3)

        # tree2=TreeNode(1)
        # tree2.left=TreeNode(2)
        # tree2.right=TreeNode(3)
        tree2=None

        print(self.isSameTree(tree1, tree2))




def main():
    sol=SameTree()
    sol.run()

if __name__ == '__main__':
    main()

